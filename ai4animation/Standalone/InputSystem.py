# Copyright (c) Meta Platforms, Inc. and affiliates.
from pathlib import Path

import pyray as pr
import raylib as rl
from ai4animation import AI4Animation, Utility
from ai4animation.Math import Rotation, Vector3

# NOTE: Gamepad name ID depends on drivers and OS
XBOX_ALIAS_1 = "xbox"
XBOX_ALIAS_2 = "x-box"
PS_ALIAS = "playstation"
THIS_DIR = Path(__file__).resolve().parent

CONTROLLER_TEXTURE = pr.load_texture(str(THIS_DIR / "resources/xbox.png"))
STICK_DEADZONE = 0.1
TRIGGER_DEADZONE = -0.9
CONTROLLER_ID = 0


def GamepadAvailable() -> bool:
    return rl.IsGamepadAvailable(CONTROLLER_ID)


def LogErrorIfGamepadNotAvailable():
    if not GamepadAvailable():
        print(f"Error: Gamepad {CONTROLLER_ID} not available")


def GetLeftStick():
    LogErrorIfGamepadNotAvailable()
    x = pr.get_gamepad_axis_movement(CONTROLLER_ID, pr.GamepadAxis.GAMEPAD_AXIS_LEFT_X)
    y = pr.get_gamepad_axis_movement(CONTROLLER_ID, pr.GamepadAxis.GAMEPAD_AXIS_LEFT_Y)
    if -STICK_DEADZONE < x < STICK_DEADZONE:
        x = 0.0
    if -STICK_DEADZONE < y < STICK_DEADZONE:
        y = 0.0
    return (x, -y)


def GetRightStick():
    LogErrorIfGamepadNotAvailable()
    x = pr.get_gamepad_axis_movement(CONTROLLER_ID, pr.GamepadAxis.GAMEPAD_AXIS_RIGHT_X)
    y = pr.get_gamepad_axis_movement(CONTROLLER_ID, pr.GamepadAxis.GAMEPAD_AXIS_RIGHT_Y)
    if -STICK_DEADZONE < x < STICK_DEADZONE:
        x = 0.0
    if -STICK_DEADZONE < y < STICK_DEADZONE:
        y = 0.0
    return (x, -y)


def GetLeftTrigger():
    LogErrorIfGamepadNotAvailable()
    value = pr.get_gamepad_axis_movement(
        CONTROLLER_ID, pr.GamepadAxis.GAMEPAD_AXIS_LEFT_TRIGGER
    )
    if value < TRIGGER_DEADZONE:
        value = -1.0
    return value


def GetRightTrigger():
    LogErrorIfGamepadNotAvailable()
    value = pr.get_gamepad_axis_movement(
        CONTROLLER_ID, pr.GamepadAxis.GAMEPAD_AXIS_RIGHT_TRIGGER
    )
    if value < TRIGGER_DEADZONE:
        value = -1.0
    return value


def IsLeftStickPressed():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_down(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_LEFT_THUMB
    )


def IsRightStickPressed():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_down(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_RIGHT_THUMB
    )

def IsL1Pressed():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_pressed(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_LEFT_TRIGGER_1
    )

def IsR1Pressed():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_pressed(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_RIGHT_TRIGGER_1
    )

def IsL2Pressed():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_pressed(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_LEFT_TRIGGER_2
    )

def IsR2Pressed():
    LogErrorIfGamepadNotAvailable()
    return pr.is_gamepad_button_pressed(
        CONTROLLER_ID, pr.GamepadButton.GAMEPAD_BUTTON_RIGHT_TRIGGER_2
    )

def GetCurrentKey():
    key = rl.GetCharPressed()
    return chr(key) if (key >= 32) and (key <= 125) else None
    # while key > 0:
    #     if (key >= 32) and (key <= 125):
    #         name += chr(key)
    #         letter_count += 1
    # key = rl.GetCharPressed()


def GetKey(id):
    return rl.IsKeyPressed(id)


def GetWASDQE():
    input = [0, 0, 0]
    if rl.IsKeyDown(rl.KEY_S):
        input[2] -= 1
    if rl.IsKeyDown(rl.KEY_W):
        input[2] += 1
    if rl.IsKeyDown(rl.KEY_A):
        input[0] -= 1
    if rl.IsKeyDown(rl.KEY_D):
        input[0] += 1
    if rl.IsKeyDown(rl.KEY_Q):
        input[1] -= 1
    if rl.IsKeyDown(rl.KEY_E):
        input[1] += 1
    return Vector3.Create(input)


# A secondary mapping to support keyboard approximation of two joysticks
def GetIJKL():
    x = 0
    y = 0
    if rl.IsKeyDown(rl.KEY_K):
        y -= 1
    if rl.IsKeyDown(rl.KEY_I):
        y += 1
    if rl.IsKeyDown(rl.KEY_J):
        x -= 1
    if rl.IsKeyDown(rl.KEY_L):
        x += 1
    return [x, y]


def GetMousePositionOnScreen():  # Get mouse position XY in screen space
    position = rl.GetMousePosition()
    return (position.x, position.y)


def GetMouseDeltaOnScreen():  # Get mouse delta XY in screen space
    position = rl.GetMouseDelta()
    return (position.x, position.y)


def GetWorldPositionOnScreen(position, camera):
    return Vector3.FromRayLib(rl.GetWorldToScreen(position.tolist(), camera))


def GetMousePositionInWorld(camera):
    ray = rl.GetScreenToWorldRay(rl.GetMousePosition(), camera)
    size = 25
    a = (-size, 0, -size)
    b = (-size, 0, size)
    c = (size, 0, size)
    d = (size, 0, -size)
    info = rl.GetRayCollisionQuad(ray, a, b, c, d)
    return Vector3.FromRayLib(info.point)


def GetMousePositionInSpace(camera, space):
    ray = rl.GetScreenToWorldRay(rl.GetMousePosition(), camera.Camera)
    size = 25
    r = camera.Entity.GetRotation()
    p = space.GetPosition()
    a = Vector3.ToRayLib(p + Rotation.Multiply(r, Vector3.Create(-size, -size, 0)))
    b = Vector3.ToRayLib(p + Rotation.Multiply(r, Vector3.Create(-size, size, 0)))
    c = Vector3.ToRayLib(p + Rotation.Multiply(r, Vector3.Create(size, size, 0)))
    d = Vector3.ToRayLib(p + Rotation.Multiply(r, Vector3.Create(size, -size, 0)))
    info = rl.GetRayCollisionQuad(ray, a, b, c, d)
    return Vector3.FromRayLib(info.point)


def DrawController(x, y, scale):
    ratio = AI4Animation.Standalone.ScaleRatio()
    x, y = AI4Animation.Standalone.ToScreen((x, y))
    w = scale * CONTROLLER_TEXTURE.width * ratio
    h = scale * CONTROLLER_TEXTURE.height * ratio
    pr.draw_texture_ex(
        CONTROLLER_TEXTURE,
        pr.Vector2(int(x - w / 2), int(y - h / 2)),
        0.0,
        scale * ratio,
        rl.DARKGRAY,
    )

    left_stick_color = pr.RED if IsLeftStickPressed() else pr.BLACK
    right_stick_color = pr.RED if IsRightStickPressed() else pr.BLACK

    # Left Stick
    pos_x = -140 * ratio
    pos_y = -73 * ratio
    outer = 40 * ratio
    middle = 30 * ratio
    inner = 20 * ratio
    stick_x, stick_y = GetLeftStick()
    pr.draw_circle(
        int(x + pos_x * scale), int(y + pos_y * scale), int(outer * scale), pr.BLACK
    )
    pr.draw_circle(
        int(x + pos_x * scale),
        int(y + pos_y * scale),
        int(middle * scale),
        pr.LIGHTGRAY,
    )
    pr.draw_circle(
        int(x + scale * (pos_x + stick_x * inner)),
        int(y + scale * (pos_y - stick_y * inner)),
        int(inner * scale),
        left_stick_color,
    )

    # Right Stick
    pos_x = 61 * ratio
    pos_y = 12 * ratio
    outer = 40 * ratio
    middle = 30 * ratio
    inner = 20 * ratio
    stick_x, stick_y = GetRightStick()
    pr.draw_circle(
        int(x + pos_x * scale), int(y + pos_y * scale), int(outer * scale), pr.BLACK
    )
    pr.draw_circle(
        int(x + pos_x * scale),
        int(y + pos_y * scale),
        int(middle * scale),
        pr.LIGHTGRAY,
    )
    pr.draw_circle(
        int(x + scale * (pos_x + stick_x * inner)),
        int(y + scale * (pos_y - stick_y * inner)),
        int(inner * scale),
        right_stick_color,
    )


def DrawWASDQE(x, y, scale):
    DrawKeySet(
        x, y, scale, [[rl.KEY_Q, rl.KEY_W, rl.KEY_E], [rl.KEY_A, rl.KEY_S, rl.KEY_D]]
    )


def DrawIJKL(x, y, scale):
    DrawKeySet(x, y, scale, [[None, rl.KEY_I, None], [rl.KEY_J, rl.KEY_K, rl.KEY_L]])


# Given a list of list of keys, draw a graphical representation of the key state
def DrawKeySet(x, y, scale, keySet):
    ratio = AI4Animation.Standalone.ScaleRatio()
    x, y = AI4Animation.Standalone.ToScreen((x, y))
    outer = 120 * ratio * scale
    spacing = 20 * ratio * scale
    border = 2

    if isinstance(keySet, int):
        keySet = [[keySet]]
    elif isinstance(keySet[0], int):
        keySet = [keySet]
    for j, row in enumerate(keySet):
        for i, key in enumerate(row):
            if key is None:
                continue
            value = rl.IsKeyDown(key)
            pos_x = x + (outer + spacing) * i
            pos_y = y + (outer + spacing) * j
            size = outer
            pr.draw_rectangle_rounded([pos_x, pos_y, size, size], 0.2, 10, pr.DARKGRAY)
            pos_x = x + (outer + spacing) * i + border
            pos_y = y + (outer + spacing) * j + border
            size = outer - 2 * border
            color = pr.LIGHTGRAY if value else pr.GRAY
            pr.draw_rectangle_rounded([pos_x, pos_y, size, size], 0.2, 10, color)
            text = Utility.ToBytes(chr(key))
            size = int(outer / 2)
            w = rl.MeasureText(text, size)
            h = size
            pos_x = x + (outer + spacing) * i + outer / 2 - w / 2
            pos_y = y + (outer + spacing) * j + outer / 2 - h / 2
            rl.DrawText(text, int(pos_x), int(pos_y), size, rl.BLACK)
