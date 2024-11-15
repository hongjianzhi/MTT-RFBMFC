from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
# from numba import jit
import copy
from .mot_online.kalman_filter import KalmanFilter
from .mot_online.basetrack import BaseTrack, TrackState
from .mot_online import matching
import cv2 as cv
import torch
def otsu_threshold(data):
    hist, _ = np.histogram(data, bins=256, range=(0, 1))
    hist = hist.astype(float) / hist.sum()
    best_threshold = 0
    best_variance = 0
    
    for threshold in np.linspace(0, 1, 256):
        background = hist[:int(threshold * 256)]
        foreground = hist[int(threshold * 256):]
        w_background = background.sum()
        w_foreground = foreground.sum()
        if w_background == 0 or w_foreground == 0:
            continue
        mean_background = np.average(np.arange(int(threshold * 256)), weights=background)
        mean_foreground = np.average(np.arange(int(threshold * 256), 256), weights=foreground)
        variance = w_background * w_foreground * ((mean_background - mean_foreground) ** 2)
        if variance > best_variance:
            best_threshold = threshold
            best_variance = variance
    
    return best_threshold
def remove_elements(array, indices):
    mask = np.ones(array.shape[0], dtype=bool)
    mask[indices] = False
    result = array[mask]
    return result

def AffineTrans(tracking_offset, stracks, mconf, image):

    RemoveRegion = []
    for SingleStrack in stracks:
        SingleStrack_tlbr = SingleStrack.tlbr.copy()
        SingleStrack_tlbr[0] = SingleStrack_tlbr[0] / image.shape[1] * tracking_offset[0].shape[2]
        SingleStrack_tlbr[1] = SingleStrack_tlbr[1] / image.shape[0] * tracking_offset[0].shape[1]
        SingleStrack_tlbr[2] = SingleStrack_tlbr[2] / image.shape[1] * tracking_offset[0].shape[2]
        SingleStrack_tlbr[3] = SingleStrack_tlbr[3] / image.shape[0] * tracking_offset[0].shape[1]
        
        SingleStrack_tlbr = (SingleStrack_tlbr+0.5).astype(int)
        
        if SingleStrack_tlbr[0] < 0:
            SingleStrack_tlbr[0] = 0
        if SingleStrack_tlbr[1] < 0:
            SingleStrack_tlbr[1] = 0
        if SingleStrack_tlbr[2] < 0:
            SingleStrack_tlbr[2] = 0
        if SingleStrack_tlbr[3] < 0:
            SingleStrack_tlbr[3] = 0
        if SingleStrack_tlbr[0] > (tracking_offset[0].shape[2] - 1):
            SingleStrack_tlbr[0] = tracking_offset[0].shape[2] - 1
        if SingleStrack_tlbr[1] > (tracking_offset[0].shape[1] - 1):
            SingleStrack_tlbr[1] = tracking_offset[0].shape[1] - 1
        if SingleStrack_tlbr[2] > (tracking_offset[0].shape[2] - 1):
            SingleStrack_tlbr[2] = tracking_offset[0].shape[2] - 1
        if SingleStrack_tlbr[3] > (tracking_offset[0].shape[1] - 1):
            SingleStrack_tlbr[3] = tracking_offset[0].shape[1] - 1
        
        for i in range(SingleStrack_tlbr[1],SingleStrack_tlbr[3]+1):
            for j in range(SingleStrack_tlbr[0],SingleStrack_tlbr[2]+1):
                RemoveRegion.append((i*tracking_offset[0].shape[2])+j)
    
    srcimg_w = np.arange(tracking_offset[0].shape[2])
    srcimg_h = np.arange(tracking_offset[0].shape[1])
    srcimg = np.array(np.meshgrid(srcimg_w,srcimg_h)).astype(float)
    srcimg[0] = (srcimg[0] / tracking_offset[0].shape[2] * image.shape[1]).astype(float)
    srcimg[1] = (srcimg[1] / tracking_offset[0].shape[1] * image.shape[0]).astype(float)

    dstimg = srcimg.copy()
    dstimg[0] = (dstimg[0] + ((tracking_offset[0][1]).cpu().numpy() / tracking_offset[0].shape[2] * image.shape[1])).astype(float)
    dstimg[1] = (dstimg[1] + ((tracking_offset[0][0]).cpu().numpy() / tracking_offset[0].shape[1] * image.shape[0])).astype(float)

    src_points = np.reshape(srcimg,(2,-1)).transpose((1,0))
    dst_points = np.reshape(dstimg,(2,-1)).transpose((1,0))

    if len(RemoveRegion) > 0:
        src_points = remove_elements(src_points, RemoveRegion)
        dst_points = remove_elements(dst_points, RemoveRegion)

    H, inliesrs = cv.estimateAffinePartial2D(src_points, dst_points, cv.RANSAC)

    return H, inliesrs

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance, self.score)

    @staticmethod

    def multi_predict(stracks, tracking_offset, mconf, image):

        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            multi_score = np.asarray([st.score for st in stracks])

            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
                    
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance, multi_score)

            H, inliesrs = AffineTrans(tracking_offset,stracks,mconf,image)
            R = H[:2,:2]
            R8x8 = np.kron(np.eye(4,dtype=float),R)
            t = H[:2,2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh),self.score)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh), new_track.score
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh),new_track.score)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class Tracker(object):
    def __init__(self, args, frame_rate=30):
        self.args = args
        self.det_thresh = args.new_thresh
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.reset()
        self.scores_his = [[0],[0],[0],[0],[0],
                           [0],[0],[0],[0],[0]]
    
    # below has no effect to final output, just to be compatible to codebase
    def init_track(self, results):
        for item in results:
            if item['score'] > self.opt.new_thresh and item['class'] == 1:
                self.id_count += 1
                item['active'] = 1
                item['age'] = 1
                item['tracking_id'] = self.id_count
                if not ('ct' in item):
                    bbox = item['bbox']
                    item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                self.tracks.append(item)
    
    def reset(self):
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.tracks = []    
        
        # below has no effect to final output, just to be compatible to codebase               
        self.id_count = 0
        
    def step(self, results, public_det=None, tracking_offset=None, mconf=None, image=None, pre_processed_images=None):

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        detections = []
        detections_second = []
        
        if len(results) == 0:
            return

        if ('is_first_frame' in pre_processed_images):
            self.scores_his = []
            self.scores_his.append(0)

        scores = np.array([item['score'] for item in results if item['class'] == 1], np.float32)
        bboxes = np.vstack([item['bbox'] for item in results if item['class'] == 1])  # N x 4, x1y1x2y2
        
        self.scores_his = np.concatenate([self.scores_his,scores],axis=0)
        self.args.track_thresh = otsu_threshold(self.scores_his)
        self.args.pre_thresh = otsu_threshold(self.scores_his)
        self.args.new_thresh = otsu_threshold(self.scores_his)

        remain_inds = scores >= self.args.track_thresh
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        
        inds_low = scores > self.args.out_thresh
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]
        
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with Kalman and IOU'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        
        STrack.multi_predict(strack_pool, tracking_offset, mconf, image)
        
        BoxPreds = []
        for TrackPred in strack_pool:
            BoxPred = TrackPred.tlbr
            BoxPreds.append(BoxPred)

        dists = matching.iou_distance(strack_pool, detections)
        
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.9)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, association the untrack to the low score detections， with IOU'''
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.4)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            #track = r_tracked_stracks[it]
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)        
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        ret = []
        for track in output_stracks:
            track_dict = {}
            track_dict['score'] = track.score
            track_dict['bbox'] = track.tlbr
            bbox = track_dict['bbox']
            track_dict['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            track_dict['active'] = 1 if track.is_activated else 0
            track_dict['tracking_id'] = track.track_id
            track_dict['class'] = 1
            ret.append(track_dict)
        
        self.tracks = ret

        return ret, BoxPreds

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
