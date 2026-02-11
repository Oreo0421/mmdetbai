# -*- coding: utf-8 -*-
"""Monkey-patch COCOeval.summarize to show 'small' and hide 'medium'/'large' for keypoints."""
import numpy as np


def patch_cocoeval_keypoint_summarize():
    """Patch xtcocotools COCOeval to show small area AP for keypoints."""
    try:
        from xtcocotools.cocoeval import COCOeval
    except ImportError:
        from pycocotools.cocoeval import COCOeval

    _orig_summarize = COCOeval.summarize

    def _new_summarize(self):
        if self.params.iouType != 'keypoints':
            return _orig_summarize(self)

        def _s(ap=1, iouThr=None, areaRng='all', maxDets=20):
            p = self.params
            aind = [i for i, a in enumerate(p.areaRngLbl) if a == areaRng]
            mind = [i for i, m in enumerate(p.maxDets) if m == maxDets]
            if ap == 1:
                s = self.eval['precision']
                if iouThr is not None:
                    t = np.where(np.isclose(p.iouThrs, iouThr))[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(np.isclose(p.iouThrs, iouThr))[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                return -1.0
            return float(np.mean(s[s > -1]))

        # 只保留 all 和 small，去掉 medium/large
        self.stats = np.zeros((8,))
        self.stats[0] = _s(1)
        self.stats[1] = _s(1, iouThr=.5)
        self.stats[2] = _s(1, iouThr=.75)
        self.stats[3] = _s(1, areaRng='small')
        self.stats[4] = _s(0)
        self.stats[5] = _s(0, iouThr=.5)
        self.stats[6] = _s(0, iouThr=.75)
        self.stats[7] = _s(0, areaRng='small')

        labels = [
            ('Precision', 'AP @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]'),
            ('Precision', 'AP @[ IoU=0.50      | area=   all | maxDets= 20 ]'),
            ('Precision', 'AP @[ IoU=0.75      | area=   all | maxDets= 20 ]'),
            ('Precision', 'AP @[ IoU=0.50:0.95 | area= small | maxDets= 20 ]'),
            ('Recall   ', 'AR @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]'),
            ('Recall   ', 'AR @[ IoU=0.50      | area=   all | maxDets= 20 ]'),
            ('Recall   ', 'AR @[ IoU=0.75      | area=   all | maxDets= 20 ]'),
            ('Recall   ', 'AR @[ IoU=0.50:0.95 | area= small | maxDets= 20 ]'),
        ]
        for i, (kind, lbl) in enumerate(labels):
            print(f' Average {kind}  ({lbl}) = {self.stats[i]:7.4f}')

    COCOeval.summarize = _new_summarize


# 导入时自动 patch
patch_cocoeval_keypoint_summarize()
