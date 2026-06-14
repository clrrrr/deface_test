import os
import numpy as np
import cv2

default_onnx_path = f'{os.path.dirname(__file__)}/scrfd_1g.onnx'


class SCRFD:
    def __init__(self, onnx_path=None, in_shape=None, backend='auto', override_execution_provider=None, gpu_id=0):
        self.in_shape = in_shape
        self.override_execution_provider = override_execution_provider  # 供多GPU初始化时复用
        self.input_size = (640, 640)  # SCRFD default input size

        if onnx_path is None:
            onnx_path = default_onnx_path

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f'SCRFD model not found: {onnx_path}')

        if backend == 'auto':
            try:
                import onnxruntime
                backend = 'onnxrt'
            except:
                backend = 'opencv'
        self.backend = backend

        if self.backend == 'opencv':
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
        elif self.backend == 'onnxrt':
            import onnxruntime
            onnxruntime.set_default_logger_severity(3)

            available_providers = onnxruntime.get_available_providers()
            if override_execution_provider is None:
                ort_providers = available_providers
            else:
                if override_execution_provider not in available_providers:
                    raise ValueError(f'{override_execution_provider=} not found. Available: {available_providers}')
                ort_providers = [override_execution_provider]

            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_profiling = False

            if 'CUDAExecutionProvider' in ort_providers:
                cuda_provider_options = {'device_id': gpu_id}
                providers_with_options = [
                    ('CUDAExecutionProvider', cuda_provider_options),
                    'CPUExecutionProvider'
                ]
            else:
                providers_with_options = ort_providers

            self.sess = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=providers_with_options
            )

            preferred_provider = self.sess.get_providers()[0]
            print(f'Running SCRFD on {preferred_provider}.')

    def __call__(self, img, threshold=0.5):
        """Detect faces in image"""
        return self.detect(img, threshold)

    def detect(self, img, threshold=0.5):
        """Detect faces and return detections"""
        h, w = img.shape[:2]

        # Prepare input
        blob, scale = self._prepare_input(img)

        # Run inference
        if self.backend == 'opencv':
            self.net.setInput(blob)
            outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        else:
            input_name = self.sess.get_inputs()[0].name
            outputs = self.sess.run(None, {input_name: blob})

        # Decode outputs
        dets = self._decode_outputs(outputs, (h, w), scale, threshold)

        # Return format: (detections, landmarks)
        # detections: [x1, y1, x2, y2, score]
        return dets, None

    def batch_call(self, imgs, threshold=0.5):
        """Batch detection for multiple images"""
        results = []
        for img in imgs:
            dets, lms = self.detect(img, threshold)
            results.append((dets, lms))
        return results

    def _prepare_input(self, img):
        """Prepare image for SCRFD input"""
        h, w = img.shape[:2]

        # Calculate scale (same as official code)
        im_ratio = float(h) / w
        model_ratio = self.input_size[1] / self.input_size[0]

        if im_ratio > model_ratio:
            new_height = self.input_size[1]
            new_width = max(1, int(new_height / im_ratio))
        else:
            new_width = self.input_size[0]
            new_height = max(1, int(new_width * im_ratio))

        scale = float(h) / new_height  # Official uses original_height / new_height

        # Resize with NEAREST (same as official)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # Pad to input_size
        det_img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized

        # Convert to blob (official: swap_rb=False, mean=(127.5,127.5,127.5), scale=1.0/128)
        blob = cv2.dnn.blobFromImage(det_img, 1.0/128.0, self.input_size, (127.5, 127.5, 127.5), swapRB=False)

        return blob, scale

    def _decode_outputs(self, outputs, img_shape, scale, threshold):
        """Decode SCRFD outputs to bounding boxes (aligned with official code)"""
        scores_list = []
        bboxes_list = []

        fmc = 3
        feat_stride_fpn = [8, 16, 32]
        num_anchors = 2

        for idx, stride in enumerate(feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + fmc] * stride  # Official multiplies by stride here

            height = self.input_size[1] // stride
            width = self.input_size[0] // stride

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))

            if num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))

            N = len(anchor_centers)
            scores = scores.reshape((N, 1))
            bbox_preds = bbox_preds.reshape((N, 4))

            # Decode boxes
            bboxes = self._distance2bbox(anchor_centers, bbox_preds)

            # Filter by threshold
            pos_inds = np.where(scores >= threshold)[0]
            if len(pos_inds) == 0:
                continue

            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])

        if len(scores_list) == 0:
            return np.empty((0, 5), dtype=np.float32)

        scores = np.vstack(scores_list)
        bboxes = np.vstack(bboxes_list)

        # Scale back to original image size
        bboxes *= scale

        # Combine to [x1, y1, x2, y2, score]
        dets = np.hstack([bboxes, scores]).astype(np.float32)

        # Sort by score
        order = scores.flatten().argsort()[::-1]
        dets = dets[order, :]

        # NMS
        keep = self._nms(dets, 0.4)
        dets = dets[keep, :]

        return dets

    def _distance2bbox(self, points, distance):
        """Convert distance to bbox"""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _nms(self, dets, iou_threshold):
        """Non-maximum suppression (aligned with official SCRFD)"""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # Official adds +1
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)  # Official adds +1
            h = np.maximum(0.0, yy2 - yy1 + 1)  # Official adds +1
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

    def shape_transform(self, in_shape, orig_shape):
        """Compatible with CenterFace interface"""
        h_orig, w_orig = orig_shape
        w_new, h_new = in_shape
        scale_w = w_new / w_orig
        scale_h = h_new / h_orig
        return w_new, h_new, scale_w, scale_h
