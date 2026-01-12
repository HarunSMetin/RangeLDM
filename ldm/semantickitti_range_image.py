from glob import glob
import os
from dataset import RangeDataset, RangeLoader, point_cloud_to_range_image
import numpy as np
import torch


class point_cloud_to_range_image_SemKITTI(point_cloud_to_range_image):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Use HDL-64E vertical profile (KITTI-360 compatible) for row assignment
        self.height = np.array(
            [0.20966667, 0.2092    , 0.2078    , 0.2078    , 0.2078    ,
             0.20733333, 0.20593333, 0.20546667, 0.20593333, 0.20546667,
             0.20453333, 0.205     , 0.2036    , 0.20406667, 0.2036    ,
             0.20313333, 0.20266667, 0.20266667, 0.20173333, 0.2008    ,
             0.2008    , 0.2008    , 0.20033333, 0.1994    , 0.20033333,
             0.19986667, 0.1994    , 0.1994    , 0.19893333, 0.19846667,
             0.19846667, 0.19846667, 0.12566667, 0.1252    , 0.1252    ,
             0.12473333, 0.12473333, 0.1238    , 0.12333333, 0.1238    ,
             0.12286667, 0.1224    , 0.12286667, 0.12146667, 0.12146667,
             0.121     , 0.12053333, 0.12053333, 0.12053333, 0.12006667,
             0.12006667, 0.1196    , 0.11913333, 0.11866667, 0.1182    ,
             0.1182    , 0.1182    , 0.11773333, 0.11726667, 0.11726667,
             0.1168    , 0.11633333, 0.11633333, 0.1154    ], dtype=np.float32)
        self.zenith = np.array([
            0.03373091,  0.02740409,  0.02276443,  0.01517224,  0.01004049,
            0.00308099, -0.00155868, -0.00788549, -0.01407172, -0.02103122,
            -0.02609267, -0.032068  , -0.03853542, -0.04451074, -0.05020488,
            -0.0565317 , -0.06180405, -0.06876355, -0.07361411, -0.08008152,
            -0.08577566, -0.09168069, -0.09793721, -0.10398284, -0.11052055,
            -0.11656618, -0.12219002, -0.12725147, -0.13407038, -0.14067839,
            -0.14510716, -0.15213696, -0.1575499 , -0.16711043, -0.17568678,
            -0.18278688, -0.19129293, -0.20247031, -0.21146846, -0.21934183,
            -0.22763699, -0.23536977, -0.24528179, -0.25477201, -0.26510582,
            -0.27326038, -0.28232882, -0.28893683, -0.30004392, -0.30953414,
            -0.31993824, -0.32816311, -0.33723155, -0.34447224, -0.352908  ,
            -0.36282001, -0.37216965, -0.38292524, -0.39164219, -0.39895318,
            -0.40703745, -0.41835542, -0.42777535, -0.43621111
        ], dtype=np.float32)
        self.incl = -self.zenith
        self.H = 64

    def get_row_inds(self, pc):
        xy_norm = np.linalg.norm(pc[:, :2], ord=2, axis=1)
        error_list = []
        for i in range(len(self.incl)):
            h = self.height[i]
            theta = self.incl[i]
            error = np.abs(theta - np.arctan2(h - pc[:, 2], xy_norm))
            error_list.append(error)
        all_error = np.stack(error_list, axis=-1)
        row_inds = np.argmin(all_error, axis=-1)
        return row_inds


class SemanticKITTIRangeDataset(RangeDataset):
    def __init__(self,
                 semantickitti_path,
                 train=True,
                 width=1024,
                 grid_sizes=[1, 1024, 1024],
                 pc_range=[-25.6, -25.6, -3., 25.6, 25.6, 1.],
                 log=False,
                 inverse=False,
                 sequences_train=("00","01","02","03","04","05","06","07","09","10"),
                 sequences_val=("08",),
                 sequences_test=("11","12","13","14","15","16","17","18","19","20","21"),
                 **kwargs):
        super().__init__(**kwargs)
        seq_root = semantickitti_path
        # Allow pointing to dataset root or sequences folder
        if os.path.isdir(os.path.join(seq_root, 'dataset', 'sequences')):
            seq_root = os.path.join(seq_root, 'dataset', 'sequences')
        elif os.path.basename(seq_root) != 'sequences':
            # Try to auto-append sequences
            seq_root = os.path.join(seq_root, 'sequences')
        self.seq_root = os.path.abspath(seq_root)
        # Cache root can be overridden by env var; defaults to ~/.cache/RangeLDM/semantickitti
        default_cache = os.path.join(os.path.expanduser('~'), '.cache', 'RangeLDM', 'semantickitti')
        self.cache_root = os.environ.get('RANGELDM_CACHE', default_cache)
        used_seqs = sequences_train if train else sequences_val
        file_list = []
        for seq in used_seqs:
            velos = sorted(glob(os.path.join(seq_root, seq, 'velodyne', '*.bin')))
            file_list.extend(velos)
        self.file_paths = file_list
        self.to_range_image = point_cloud_to_range_image_SemKITTI(
            width=width,
            grid_sizes=grid_sizes,
            pc_range=pc_range,
            log=log,
            inverse=inverse,
        )

    def get_pts(self, pts_path):
        return np.fromfile(pts_path, dtype=np.float32).reshape(-1, 4)

    def get_pth_path(self, pts_path):
        # Cache in a writable mirror directory rather than inside dataset tree (which may be read-only).
        # Structure: <cache_root>/<relative-to-seq-root>/ with velodyne -> velodyne_range.
        rel = os.path.relpath(pts_path, start=self.seq_root)
        rel = rel.replace(os.sep + 'velodyne' + os.sep, os.sep + 'velodyne_range' + os.sep)
        rel = rel[:-4] + '.pth' if rel.endswith('.bin') else rel + '.pth'
        return os.path.join(self.cache_root, rel)


class SemanticKITTIRangeLoader(RangeLoader):
    def __init__(self,
                 semantickitti_path,
                 used_feature=2,
                 width=1024,
                 grid_sizes=[1, 1024, 1024],
                 pc_range=[-25.6, -25.6, -3., 25.6, 25.6, 1.],
                 log=False,
                 inverse=False,
                 downsample=None,
                 inpainting=None,
                 coord=False,
                 **kwargs):
        super().__init__(**kwargs)
        train_dataset = SemanticKITTIRangeDataset(
            semantickitti_path,
            train=True,
            used_feature=used_feature,
            width=width,
            grid_sizes=grid_sizes,
            pc_range=pc_range,
            log=log,
            inverse=inverse,
            downsample=downsample,
            inpainting=inpainting,
            coord=coord,
        )
        # Validation on sequence 08 by default
        test_dataset = SemanticKITTIRangeDataset(
            semantickitti_path,
            train=False,
            used_feature=used_feature,
            width=width,
            grid_sizes=grid_sizes,
            pc_range=pc_range,
            log=log,
            inverse=inverse,
            downsample=downsample,
            inpainting=inpainting,
            coord=coord,
        )
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
