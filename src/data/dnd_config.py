
DND_JOINT_NAMES = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Spine4', 'Neck', 'Head', 'Head_end'
              , 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'
              , 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb3_end'
              , 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex3_end'
              , 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle3_end'
              , 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing3_end'
              , 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky3_end'
              , 'LeftHand_end'
              , 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'
              , 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb3_end'
              , 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex3_end'
              , 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle3_end'
              , 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing3_end'
              , 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky3_end'
              , 'RightHand_end'
              , 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToeBase_end'
              , 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToeBase_end']

HAND_JOINTS = [  'LeftHand'
              , 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb3_end'
              , 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex3_end'
              , 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle3_end'
              , 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing3_end'
              , 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky3_end'
              , 'LeftHand_end'
              , 'RightHand'
              , 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb3_end'
              , 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex3_end'
              , 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle3_end'
              , 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing3_end'
              , 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky3_end'
              , 'RightHand_end']

LEFT_HAND_JOINTS = [  'LeftHand'
              , 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb3_end'
              , 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex3_end'
              , 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle3_end'
              , 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing3_end'
              , 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky3_end'
              , 'LeftHand_end']

RIGHT_HAND_JOINTS = [ 'RightHand'
              , 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb3_end'
              , 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex3_end'
              , 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle3_end'
              , 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing3_end'
              , 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky3_end'
              , 'RightHand_end']

    
BODY_JOINTS = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Spine4', 'Neck', 'Head', 'Head_end'
              , 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'
              , 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'
              , 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToeBase_end'
              , 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToeBase_end'
              ]

dnd_body_idxs = [DND_JOINT_NAMES.index(joint) for joint in BODY_JOINTS]
dnd_hand_idxs = [DND_JOINT_NAMES.index(joint) for joint in HAND_JOINTS]
dnd_right_hand_idxs = [DND_JOINT_NAMES.index(joint) for joint in RIGHT_HAND_JOINTS]
dnd_left_hand_idxs = [DND_JOINT_NAMES.index(joint) for joint in LEFT_HAND_JOINTS]
body_wrist_r = BODY_JOINTS.index('RightHand')
body_wrist_l = BODY_JOINTS.index('LeftHand')   
    
DND_JOINT_CHAINS =  [['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Spine4', 'Neck', 'Head', 'Head_end'], 
                    ['Spine3', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHand_end'], 
                    ['LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb3_end'],
                    ['LeftHand', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex3_end'],
                    ['LeftHand', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle3_end'],
                    ['LeftHand', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing3_end'],
                    ['LeftHand', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky3_end'],
                    ['Spine3', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHand_end'],
                    ['RightHand', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb3_end'], 
                    ['RightHand', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex3_end'],
                    ['RightHand', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle3_end'], 
                    ['RightHand', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing3_end'], 
                    ['RightHand', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky3_end'], 
                    ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToeBase_end'], 
                    ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToeBase_end']]


DND_JOINT_CHAINS_IDX = [ [ DND_JOINT_NAMES.index(joint) for joint in chain ] for chain in DND_JOINT_CHAINS]


DND_JOINT_PARENT =  {'Hips': 'None',
 'Spine': 'Hips',
 'Spine1': 'Spine',
 'Spine2': 'Spine1',
 'Spine3': 'Spine2',
 'Spine4': 'Spine3',
 'Neck': 'Spine4',
 'Head': 'Neck',
 'Head_end': 'Head',
 'LeftShoulder': 'Spine3',
 'LeftArm': 'LeftShoulder',
 'LeftForeArm': 'LeftArm',
 'LeftHand': 'LeftForeArm',
 'LeftHandThumb1': 'LeftHand',
 'LeftHandThumb2': 'LeftHandThumb1',
 'LeftHandThumb3': 'LeftHandThumb2',
 'LeftHandThumb3_end': 'LeftHandThumb3',
 'LeftHandIndex1': 'LeftHand',
 'LeftHandIndex2': 'LeftHandIndex1',
 'LeftHandIndex3': 'LeftHandIndex2',
 'LeftHandIndex3_end': 'LeftHandIndex3',
 'LeftHandMiddle1': 'LeftHand',
 'LeftHandMiddle2': 'LeftHandMiddle1',
 'LeftHandMiddle3': 'LeftHandMiddle2',
 'LeftHandMiddle3_end': 'LeftHandMiddle3',
 'LeftHandRing1': 'LeftHand',
 'LeftHandRing2': 'LeftHandRing1',
 'LeftHandRing3': 'LeftHandRing2',
 'LeftHandRing3_end': 'LeftHandRing3',
 'LeftHandPinky1': 'LeftHand',
 'LeftHandPinky2': 'LeftHandPinky1',
 'LeftHandPinky3': 'LeftHandPinky2',
 'LeftHandPinky3_end': 'LeftHandPinky3',
 'LeftHand_end': 'LeftHand',
 'RightShoulder': 'Spine3',
 'RightArm': 'RightShoulder',
 'RightForeArm': 'RightArm',
 'RightHand': 'RightForeArm',
 'RightHandThumb1': 'RightHand',
 'RightHandThumb2': 'RightHandThumb1',
 'RightHandThumb3': 'RightHandThumb2',
 'RightHandThumb3_end': 'RightHandThumb3',
 'RightHandIndex1': 'RightHand',
 'RightHandIndex2': 'RightHandIndex1',
 'RightHandIndex3': 'RightHandIndex2',
 'RightHandIndex3_end': 'RightHandIndex3',
 'RightHandMiddle1': 'RightHand',
 'RightHandMiddle2': 'RightHandMiddle1',
 'RightHandMiddle3': 'RightHandMiddle2',
 'RightHandMiddle3_end': 'RightHandMiddle3',
 'RightHandRing1': 'RightHand',
 'RightHandRing2': 'RightHandRing1',
 'RightHandRing3': 'RightHandRing2',
 'RightHandRing3_end': 'RightHandRing3',
 'RightHandPinky1': 'RightHand',
 'RightHandPinky2': 'RightHandPinky1',
 'RightHandPinky3': 'RightHandPinky2',
 'RightHandPinky3_end': 'RightHandPinky3',
 'RightHand_end': 'RightHand',
 'LeftUpLeg': 'Hips',
 'LeftLeg': 'LeftUpLeg',
 'LeftFoot': 'LeftLeg',
 'LeftToeBase': 'LeftFoot',
 'LeftToeBase_end': 'LeftToeBase',
 'RightUpLeg': 'Hips',
 'RightLeg': 'RightUpLeg',
 'RightFoot': 'RightLeg',
 'RightToeBase': 'RightFoot',
 'RightToeBase_end': 'RightToeBase'}

ROT_DND_JOINT_NAMES = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Spine4', 'Neck', 'Head'
              , 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'
              , 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3'
              , 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3'
              , 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3'
              , 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3'
              , 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3'
              
              , 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'
              , 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3'
              , 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3'
              , 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3'
              , 'RightHandRing1', 'RightHandRing2', 'RightHandRing3'
              , 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3'
              
              , 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'
              , 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']

ROT_HAND_JOINTS = [  'LeftHand'
              , 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3'
              , 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3'
              , 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3'
              , 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3'
              , 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3'
              
              , 'RightHand'
              , 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3'
              , 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3'
              , 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3'
              , 'RightHandRing1', 'RightHandRing2', 'RightHandRing3'
              , 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3'
              ]

ROT_LEFT_HAND_JOINTS = [  'LeftHand'
              , 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3'
              , 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3'
              , 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3'
              , 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3'
              , 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3'
              ]

ROT_RIGHT_HAND_JOINTS = [ 'RightHand'
              , 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3'
              , 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3'
              , 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3'
              , 'RightHandRing1', 'RightHandRing2', 'RightHandRing3'
              , 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3'
              ]

    
ROT_BODY_JOINTS = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Spine4', 'Neck', 'Head'
              , 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'
              , 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'
              , 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'
              , 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'
              ]

rot_dnd_body_idxs = [ROT_DND_JOINT_NAMES.index(joint) for joint in ROT_BODY_JOINTS]
rot_dnd_hand_idxs = [ROT_DND_JOINT_NAMES.index(joint) for joint in ROT_HAND_JOINTS]
rot_dnd_right_hand_idxs = [ROT_DND_JOINT_NAMES.index(joint) for joint in ROT_RIGHT_HAND_JOINTS]
rot_dnd_left_hand_idxs = [ROT_DND_JOINT_NAMES.index(joint) for joint in ROT_LEFT_HAND_JOINTS]
rot_body_wrist_r = ROT_BODY_JOINTS.index('RightHand')
rot_body_wrist_l = ROT_BODY_JOINTS.index('LeftHand')   