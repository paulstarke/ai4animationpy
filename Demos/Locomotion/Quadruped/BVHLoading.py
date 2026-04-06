# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

from ai4animation import Actor, AI4Animation, Motion, Time, Tensor, Transform, Vector3

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Quadruped")

sys.path.append(ASSETS_PATH)
import Definitions

JOINT_CORRECTIONS = {
    Definitions.HeadName: Vector3.Create(90.0, 0.0, 0.0),
    Definitions.LeftShoulderName: Vector3.Create(90.0, 0.0, 0.0),
    Definitions.RightShoulderName: Vector3.Create(90.0, 0.0, 0.0),
}


class Program:
    def __init__(self, path):
        self.Path = path

    def Start(self):
        self.Motion = Motion.LoadFromBVH(
            self.Path,
            scale=0.01,
            names=Definitions.FULL_BODY_NAMES,
            mirror_axis=Vector3.Axis.XPositive,
            joint_corrections=JOINT_CORRECTIONS,
        )
        self.Mirror = False

        self.Pose = None

        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(
            Actor,
            os.path.join(ASSETS_PATH, "Wolf.glb"),
            Definitions.FULL_BODY_NAMES,
        )

    def Update(self):
        timestamp = Time.TotalTime % self.Motion.TotalTime
        self.Pose = self.Motion.GetBoneTransformations(
            timestamps=timestamp, mirrored=self.Mirror
        )
        self.Actor.SetTransforms(
            self.Motion.GetBoneTransformations(
                timestamps=timestamp,
                bone_names_or_indices=self.Actor.GetBoneNames(),
                mirrored=self.Mirror,
            )
        )
        self.Actor.SyncToScene()

    # def GUI(self):
    #     AI4Animation.Draw.Text3D(self.Motion.Hierarchy.BoneNames, Transform.GetPosition(self.Pose), size=0.0125, color=AI4Animation.Color.BLACK)

    # def Draw(self):
    #     AI4Animation.Draw.Transform(self.Pose, size=0.5, axisSize=0.25)


def main():
    AI4Animation(Program("D1_001_KAN01_001.bvh"))


if __name__ == "__main__":
    main()
