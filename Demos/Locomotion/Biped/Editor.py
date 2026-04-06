# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
from pathlib import Path

from ai4animation import (
    AI4Animation,
    ContactModule,
    Dataset,
    MotionEditor,
    MotionModule,
    GuidanceModule,
    RootModule,
)

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent.parent / "_ASSETS_/Geno")

sys.path.append(ASSETS_PATH)
import Definitions


class Program:
    def Start(self):
        editor = AI4Animation.Scene.AddEntity("MotionEditor")
        self.dataset = Dataset(
                os.path.join(ASSETS_PATH, "Mocap", "style100"),
                [
                    lambda x: RootModule(
                        x,
                        Definitions.HipName,
                        Definitions.LeftHipName,
                        Definitions.RightHipName,
                        Definitions.LeftShoulderName,
                        Definitions.RightShoulderName,
                        Definitions.NeckName,
                    ),
                    lambda x: MotionModule(x),
                    lambda x: ContactModule(
                        x,
                        [
                            (Definitions.LeftAnkleName, 0.1, 0.25),
                            (Definitions.LeftBallName, 0.05, 0.25),
                            (Definitions.RightAnkleName, 0.1, 0.25),
                            (Definitions.RightBallName, 0.05, 0.25),
                        ],
                    ),
                    lambda x: GuidanceModule(x),
                ],
            )
        bones = Definitions.FULL_BODY_NAMES
        editor.AddComponent(
            MotionEditor,
            self.dataset,
            os.path.join(ASSETS_PATH, "Model.glb"),
            bones
        )

        AI4Animation.Standalone.Camera.SetTarget(editor)

    def Update(self):
        pass


def main():
    AI4Animation(Program())


if __name__ == "__main__":
    main()
