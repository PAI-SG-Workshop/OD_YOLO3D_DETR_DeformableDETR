"""Train yolo3d deformable_detr (DETR-family KITTI BEV track), then evaluate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.deformable_detr.build import build_model

if __name__ == "__main__":
    main("train", task="yolo3d", version="deformable_detr", build_model=build_model)
