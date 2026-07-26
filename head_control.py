"""
Contrôle bas niveau de la tête et des yeux de Reachy 2019.

Ce fichier est une reconstruction compatible avec
kamel_bouabcha_reachy_code.py.

Configuration connue du robot :
- Dynamixel SDK, protocole 1.0
- Port : /dev/ttyUSB0
- Baudrate : 1 000 000
- Tête horizontale : ID 0
- Tête verticale : ID 2
- Œil vertical : ID 5
- Œil horizontal : ID 6

Important :
- ce module ne déplace aucun moteur automatiquement à l'import ;
- Robot_Interface() ouvre le port et active le couple ;
- les mouvements sont limités à des plages prudentes ;
- si un axe part dans le mauvais sens, utiliser EYE_INVERT_X/Y dans
  le code principal plutôt que modifier ce fichier immédiatement.
"""

from __future__ import annotations

import time
from typing import Final

try:
    from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
except ImportError as exc:
    raise ImportError(
        "Le paquet dynamixel_sdk est absent. Installe-le avec :\n"
        "pip install dynamixel-sdk"
    ) from exc


# -----------------------------------------------------------------------------
# Communication Dynamixel
# -----------------------------------------------------------------------------
PROTOCOL_VERSION: Final[float] = 1.0
BAUDRATE: Final[int] = 1_000_000
DEVICENAME: Final[str] = "/dev/ttyUSB0"

# Table de contrôle Dynamixel MX, protocole 1.0.
ADDR_MX_TORQUE_ENABLE: Final[int] = 24
ADDR_MX_GOAL_POSITION: Final[int] = 30
ADDR_MX_MOVING_SPEED: Final[int] = 32
ADDR_MX_PRESENT_POSITION: Final[int] = 36

TORQUE_ENABLE: Final[int] = 1
TORQUE_DISABLE: Final[int] = 0


# -----------------------------------------------------------------------------
# Identifiants moteurs du Reachy 2019 du laboratoire
# -----------------------------------------------------------------------------
BLOW_HORIZONTAL: Final[int] = 0
BLOW_VERTICAL: Final[int] = 2
EYE_VERTICAL: Final[int] = 5
EYE_HORIZONTAL: Final[int] = 6

MOTOR_IDS: Final[tuple[int, ...]] = (
    BLOW_HORIZONTAL,
    BLOW_VERTICAL,
    EYE_VERTICAL,
    EYE_HORIZONTAL,
)

# Ordre historique :
# [tête horizontale, tête verticale, œil vertical, œil horizontal]
INIT_POS: Final[list[int]] = [2050, 2000, 2075, 3130]

# Plage horizontale retrouvée dans l'ancien fichier.
EYE_HORIZONTAL_RANGE: Final[list[int]] = [3270, 2900]

# La valeur exacte de l'ancienne plage verticale n'a pas été retrouvée.
# Cette plage est volontairement prudente autour de INIT_POS[2].
EYE_VERTICAL_RANGE: Final[list[int]] = [2250, 1900]

# Limites cohérentes avec le code principal :
# HEAD_H_LIMIT = 220 et HEAD_V_LIMIT = 80.
HEAD_HORIZONTAL_RANGE: Final[list[int]] = [
    INIT_POS[0] + 220,
    INIT_POS[0] - 220,
]
HEAD_VERTICAL_RANGE: Final[list[int]] = [
    INIT_POS[1] + 80,
    INIT_POS[1] - 80,
]

DEFAULT_EYE_SPEED: Final[int] = 60


def _ordered_limits(values: list[int]) -> tuple[int, int]:
    """Retourne toujours (minimum, maximum), quel que soit l'ordre donné."""
    return min(int(values[0]), int(values[1])), max(int(values[0]), int(values[1]))


EYE_H_MIN, EYE_H_MAX = _ordered_limits(EYE_HORIZONTAL_RANGE)
EYE_V_MIN, EYE_V_MAX = _ordered_limits(EYE_VERTICAL_RANGE)
HEAD_H_MIN, HEAD_H_MAX = _ordered_limits(HEAD_HORIZONTAL_RANGE)
HEAD_V_MIN, HEAD_V_MAX = _ordered_limits(HEAD_VERTICAL_RANGE)

MOTOR_LIMITS: Final[dict[int, tuple[int, int]]] = {
    BLOW_HORIZONTAL: (HEAD_H_MIN, HEAD_H_MAX),
    BLOW_VERTICAL: (HEAD_V_MIN, HEAD_V_MAX),
    EYE_VERTICAL: (EYE_V_MIN, EYE_V_MAX),
    EYE_HORIZONTAL: (EYE_H_MIN, EYE_H_MAX),
}


class Robot_Interface:
    """Interface minimale utilisée par le pipeline principal de Kamel."""

    def __init__(
        self,
        device_name: str = DEVICENAME,
        baudrate: int = BAUDRATE,
    ) -> None:
        self.device_name = str(device_name)
        self.baudrate = int(baudrate)
        self.port_handler = PortHandler(self.device_name)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        self.closed = False

        if not self.port_handler.openPort():
            raise RuntimeError(
                f"Impossible d'ouvrir {self.device_name}. "
                "Vérifie le câble, ferme les autres scripts et les programmes "
                "qui utilisent /dev/ttyUSB0."
            )

        if not self.port_handler.setBaudRate(self.baudrate):
            self.port_handler.closePort()
            raise RuntimeError(
                f"Impossible de régler le baudrate à {self.baudrate}."
            )

        print(
            f"[DYNAMIXEL] Port ouvert : {self.device_name} | "
            f"baudrate={self.baudrate} | protocole={PROTOCOL_VERSION}"
        )

        try:
            for motor_id in MOTOR_IDS:
                self.enable_torque(motor_id)

            # Vitesse prudente des deux axes des yeux.
            self.set_speed(EYE_HORIZONTAL, DEFAULT_EYE_SPEED)
            self.set_speed(EYE_VERTICAL, DEFAULT_EYE_SPEED)
        except Exception:
            self._close_port_only()
            raise

    # ------------------------------------------------------------------
    # Vérification des réponses SDK
    # ------------------------------------------------------------------
    def _check_result(
        self,
        communication_result: int,
        packet_error: int,
        action: str,
        motor_id: int,
    ) -> None:
        if communication_result != COMM_SUCCESS:
            detail = self.packet_handler.getTxRxResult(communication_result)
            raise RuntimeError(
                f"[DYNAMIXEL] {action} impossible sur ID {motor_id} : {detail}"
            )

        if packet_error != 0:
            detail = self.packet_handler.getRxPacketError(packet_error)
            raise RuntimeError(
                f"[DYNAMIXEL] Erreur moteur ID {motor_id} pendant {action} : {detail}"
            )

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("L'interface Dynamixel est déjà fermée.")

    # ------------------------------------------------------------------
    # Couple
    # ------------------------------------------------------------------
    def enable_torque(self, motor_id: int) -> None:
        self._ensure_open()
        communication_result, packet_error = (
            self.packet_handler.write1ByteTxRx(
                self.port_handler,
                int(motor_id),
                ADDR_MX_TORQUE_ENABLE,
                TORQUE_ENABLE,
            )
        )
        self._check_result(
            communication_result,
            packet_error,
            "activation du couple",
            int(motor_id),
        )

    def disable_torque(self, motor_id: int) -> None:
        self._ensure_open()
        communication_result, packet_error = (
            self.packet_handler.write1ByteTxRx(
                self.port_handler,
                int(motor_id),
                ADDR_MX_TORQUE_ENABLE,
                TORQUE_DISABLE,
            )
        )
        self._check_result(
            communication_result,
            packet_error,
            "désactivation du couple",
            int(motor_id),
        )

    # ------------------------------------------------------------------
    # Lecture et écriture génériques
    # ------------------------------------------------------------------
    def get_position(self, motor_id: int) -> int:
        self._ensure_open()
        position, communication_result, packet_error = (
            self.packet_handler.read2ByteTxRx(
                self.port_handler,
                int(motor_id),
                ADDR_MX_PRESENT_POSITION,
            )
        )
        self._check_result(
            communication_result,
            packet_error,
            "lecture de position",
            int(motor_id),
        )
        return int(position)

    def set_position(self, motor_id: int, position: int) -> int:
        """Envoie une position en respectant les limites connues du moteur."""
        self._ensure_open()
        motor_id = int(motor_id)
        position = int(position)

        limits = MOTOR_LIMITS.get(motor_id)
        if limits is not None:
            minimum, maximum = limits
            position = max(minimum, min(position, maximum))
        else:
            position = max(0, min(position, 4095))

        communication_result, packet_error = (
            self.packet_handler.write2ByteTxRx(
                self.port_handler,
                motor_id,
                ADDR_MX_GOAL_POSITION,
                position,
            )
        )
        self._check_result(
            communication_result,
            packet_error,
            "écriture de position",
            motor_id,
        )
        return position

    def set_speed(self, motor_id: int, speed: int) -> int:
        """Règle une vitesse Dynamixel comprise entre 1 et 1023."""
        self._ensure_open()
        motor_id = int(motor_id)
        speed = max(1, min(int(speed), 1023))

        communication_result, packet_error = (
            self.packet_handler.write2ByteTxRx(
                self.port_handler,
                motor_id,
                ADDR_MX_MOVING_SPEED,
                speed,
            )
        )
        self._check_result(
            communication_result,
            packet_error,
            "réglage de vitesse",
            motor_id,
        )
        return speed

    # ------------------------------------------------------------------
    # Déplacements relatifs des yeux
    # ------------------------------------------------------------------
    def _move_relative(self, motor_id: int, delta: int) -> int:
        current = self.get_position(motor_id)
        return self.set_position(motor_id, current + int(delta))

    def go_left(self, step: int = 2) -> int:
        # Sur ce montage, la gauche correspond normalement à une augmentation.
        return self._move_relative(EYE_HORIZONTAL, abs(int(step)))

    def go_right(self, step: int = 2) -> int:
        return self._move_relative(EYE_HORIZONTAL, -abs(int(step)))

    def go_up(self, step: int = 2) -> int:
        # Si le sens vertical est inversé, activer EYE_INVERT_Y dans le main.
        return self._move_relative(EYE_VERTICAL, abs(int(step)))

    def go_down(self, step: int = 2) -> int:
        return self._move_relative(EYE_VERTICAL, -abs(int(step)))

    # ------------------------------------------------------------------
    # Fonctions de compatibilité avec les anciens scripts
    # ------------------------------------------------------------------
    def initial_pos(self, delay: float = 0.8) -> None:
        """Replace explicitement la tête et les yeux aux positions historiques."""
        self.set_speed(BLOW_HORIZONTAL, 50)
        self.set_speed(BLOW_VERTICAL, 50)
        self.set_speed(EYE_VERTICAL, DEFAULT_EYE_SPEED)
        self.set_speed(EYE_HORIZONTAL, DEFAULT_EYE_SPEED)

        self.set_position(BLOW_HORIZONTAL, INIT_POS[0])
        self.set_position(BLOW_VERTICAL, INIT_POS[1])
        self.set_position(EYE_VERTICAL, INIT_POS[2])
        self.set_position(EYE_HORIZONTAL, INIT_POS[3])
        time.sleep(max(0.0, float(delay)))

    def test(self, step: int = 4, delay: float = 0.25) -> None:
        """Petit test prudent des yeux, avec retour approximatif au centre."""
        step = max(1, min(abs(int(step)), 8))
        delay = max(0.05, float(delay))

        self.go_left(step)
        time.sleep(delay)
        self.go_right(step * 2)
        time.sleep(delay)
        self.go_left(step)
        time.sleep(delay)

        self.go_up(step)
        time.sleep(delay)
        self.go_down(step * 2)
        time.sleep(delay)
        self.go_up(step)
        time.sleep(delay)

    # ------------------------------------------------------------------
    # Fermeture
    # ------------------------------------------------------------------
    def _close_port_only(self) -> None:
        if not self.closed:
            try:
                self.port_handler.closePort()
            finally:
                self.closed = True

    def shutdown(self, disable_torque: bool = True) -> None:
        """Ferme proprement la communication.

        Par défaut, le couple est désactivé comme dans les anciens scripts.
        Attention : la tête peut devenir libre lorsque le couple est coupé.
        """
        if self.closed:
            return

        if disable_torque:
            for motor_id in reversed(MOTOR_IDS):
                try:
                    self.disable_torque(motor_id)
                except Exception as exc:
                    print(f"[DYNAMIXEL] Avertissement à l'arrêt, ID {motor_id} : {exc}")

        self._close_port_only()
        print("[DYNAMIXEL] Port fermé.")

    def __enter__(self) -> "Robot_Interface":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown()


if __name__ == "__main__":
    print("Test direct prudent de head_control.py")
    print("Ctrl+C pour arrêter.")
    robot = Robot_Interface()
    try:
        print("Positions actuelles :")
        for motor_id in MOTOR_IDS:
            print(f"  ID {motor_id}: {robot.get_position(motor_id)}")
        robot.test(step=3)
    finally:
        robot.shutdown()
