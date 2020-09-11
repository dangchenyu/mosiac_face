# Mosaic Face


## Introduction
Mosaic pedestrians' faces for protecting their pravicy.


## Usage
Run test.py and change parameters of save_folder, video_folder, and video_output with your own configuration. 

Arguments are listed below:
parser.add_argument('--trained_model', default='weights/light_DSFD.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval_tools/light_DSFD/', type=str,
                    help='mosaiced img saving folder')
parser.add_argument('--visual_threshold', default=0.9, type=float,
                    help='Final confidence threshold')
parser.add_argument('--area_scale', default=1.25, type=float,
                    help='scale of mosaic area')
parser.add_argument('--cuda', default=True, type=bool,
                    help='If use cuda')
parser.add_argument('--video_folder', default='', type=str,
                    help='origin video folder')
parser.add_argument('--widerface_root', default=WIDERFace_ROOT, help='Location of VOC root directory')
parser.add_argument('--video_output', default='/home/rvlab/Desktop/', type=str,
                    help='processed video folder ')

