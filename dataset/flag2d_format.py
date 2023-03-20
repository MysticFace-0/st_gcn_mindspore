"""
dataset{
split: # divide dataset to 'train' and 'val' according to identifiers
    {
    train: #S:001 C:001,002,006 M:001-004 P:001-010 A:001-060 R:001-003     len:21600
        ['S001C001M001P001A001R001', 'S001C001M001P001A001R002', 'S001C001M001P001A001R003', ...],
    val: # PLAY,PHR,PLJP,PLRH,PCSH,PCYJ,PWYQ,PZYX,PTYZ,PLZH  A:001-060 V001,002 R:001-003 S:001,002     len:7200
        ['PLAY_A001_V001_R001_S001', 'PLAY_A001_V001_R001_S002', 'PLAY_A001_V001_R002_S001', ...]
    },
annotations:
    [
    '''
    keypoint:
    1 1045 17 2
    1 1005 17 2
    1 1085 17 2
    1 1286 17 2
    1 1023 17 2
    1 1034 17 2
    1 971 17 2
    1 1004 17 2
    ...
    '''
    {'frame_dir': 'S001C001M002P005A015R003', 'label': 14, 'img_shape': (480, 854), 'original_shape': (480, 854), 'total_frames': 1934,
    'keypoint': array([[[[430. , 163.5],
         [434.2, 159.2],
         [425.8, 159.2],
         ...,
         [415.2, 294.5],
         [440.8, 339. ],
         [413.2, 339. ]],

        [[430.2, 161.4],
         [434.2, 159.2],
         [426. , 159.2],
         ...,
         [415.2, 294.5],
         [440.8, 339. ],
         [413.2, 339. ]],

        [[429.8, 161.4],
         [434. , 159.2],
         [427.8, 157.1],
         ...,
         [415. , 294.8],
         [440.2, 339. ],
         [412.8, 339. ]],

        ...,

        [[429.2, 163.1],
         [433.2, 158.9],
         [427. , 158.9],
         ...,
         [416.5, 294.8],
         [441.8, 339.5],
         [416.5, 339.5]],

        [[429.2, 163. ],
         [433.5, 158.8],
         [427.2, 158.8],
         ...,
         [416.8, 294.5],
         [440. , 339.2],
         [416.8, 339.2]],

        [[429.5, 163. ],
         [433.8, 160.9],
         [427.5, 158.8],
         ...,
         [416.8, 294.8],
         [440.2, 339.5],
         [416.8, 339.5]]]], dtype=float16),
         'keypoint_score': array([[[0.9575, 0.958 , 0.9834, ..., 0.92  , 0.912 , 0.9146],
        [0.9575, 0.9556, 0.983 , ..., 0.9204, 0.915 , 0.913 ],
        [0.9624, 0.966 , 0.982 , ..., 0.9175, 0.8955, 0.9116],
        ...,
        [0.975 , 0.9863, 0.9795, ..., 0.93  , 0.8735, 0.9062],
        [0.967 , 0.978 , 0.9775, ..., 0.933 , 0.8755, 0.9062],
        [0.9663, 0.975 , 0.979 , ..., 0.9224, 0.8745, 0.9   ]]],
      dtype=float16)},
      {'frame_dir': 'S001C001M002P005A016R001', 'label': 15, 'img_shape': (480, 854), 'original_shape': (480, 854), 'total_frames': 1435,
      'keypoint': array([[[[428.2, 162.9],
         [430.5, 158.6],
         [424. , 158.6],
         ...,
         [413.5, 294. ],
         [441. , 338.2],
         [413.5, 336.2]],

        [[428. , 162.9],
         [432.2, 158.6],
         [426. , 158.6],
         ...,
         [413.2, 294.2],
         [440.8, 338.5],
         [413.2, 336.5]],

        [[427.8, 163. ],
         [432. , 158.8],
         [425.8, 158.8],
         ...,
         [413. , 294.2],
         [440.5, 338.8],
         [413. , 336.5]],

        ...,

        [[430.2, 161.8],
         [434.5, 157.6],
         [428.2, 157.6],
         ...,
         [415.8, 293. ],
         [440.8, 334.8],
         [415.8, 334.8]],

        [[430. , 161.9],
         [434.2, 157.8],
         [428. , 157.8],
         ...,
         [417.5, 293.2],
         [440.5, 334.8],
         [417.5, 334.8]],

        [[430. , 161.9],
         [434.2, 157.8],
         [428. , 157.8],
         ...,
         [417.5, 293. ],
         [440.5, 334.8],
         [417.5, 334.8]]]], dtype=float16),
         'keypoint_score': array([[[0.9336, 0.9404, 0.988 , ..., 0.898 , 0.8755, 0.892 ],
        [0.943 , 0.957 , 0.988 , ..., 0.8955, 0.882 , 0.886 ],
        [0.9355, 0.956 , 0.982 , ..., 0.893 , 0.8813, 0.8853],
        ...,
        [0.945 , 0.9434, 0.9307, ..., 0.929 , 0.9214, 0.921 ],
        [0.933 , 0.9443, 0.9595, ..., 0.9194, 0.924 , 0.919 ],
        [0.941 , 0.9517, 0.957 , ..., 0.9204, 0.9224, 0.92  ]]],
      dtype=float16)},

    ...

    ]
}

"""