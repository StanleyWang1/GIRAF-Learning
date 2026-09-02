from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler


class DynamixelController:
    """Thin wrapper over the Dynamixel SDK port and packet handlers.

    Commands are ``(address, byte_length)`` tuples from ``dynamixel_config``.
    """

    def __init__(self, device_name, baudrate, protocol_version):
        self.device_name = device_name
        self.baudrate = baudrate
        self.protocol_version = protocol_version
        self.port_handler = PortHandler(self.device_name)
        self.packet_handler = PacketHandler(self.protocol_version)
        self.open_port()
        self.set_baudrate()

    def open_port(self):
        if not self.port_handler.openPort():
            raise RuntimeError("Failed to open the port")

    def set_baudrate(self):
        if not self.port_handler.setBaudRate(self.baudrate):
            raise RuntimeError("Failed to change the baudrate")

    def _check(self, dxl_id, comm_result, error):
        if comm_result != COMM_SUCCESS:
            print(
                f"Communication error on motor {dxl_id}: "
                f"{self.packet_handler.getTxRxResult(comm_result)}"
            )
            return False
        if error != 0:
            print(
                f"Packet error on motor {dxl_id}: "
                f"{self.packet_handler.getRxPacketError(error)}"
            )
            return False
        return True

    def WRITE(self, dxl_id, command_type, command_value):
        """Write ``command_value`` to a motor. Returns True on success."""
        address, length = command_type
        writers = {
            1: self.packet_handler.write1ByteTxRx,
            2: self.packet_handler.write2ByteTxRx,
            4: self.packet_handler.write4ByteTxRx,
        }
        if length not in writers:
            print(f"Invalid byte length: {length}")
            return False
        comm_result, error = writers[length](
            self.port_handler, dxl_id, address, command_value
        )
        return self._check(dxl_id, comm_result, error)

    def READ(self, dxl_id, command_type):
        """Read a value from a motor. Returns the value, or False on failure."""
        address, length = command_type
        readers = {
            1: self.packet_handler.read1ByteTxRx,
            2: self.packet_handler.read2ByteTxRx,
            4: self.packet_handler.read4ByteTxRx,
        }
        if length not in readers:
            print(f"Invalid byte length: {length}")
            return False
        value, comm_result, error = readers[length](self.port_handler, dxl_id, address)
        if not self._check(dxl_id, comm_result, error):
            return False
        return value

    def close_port(self):
        self.port_handler.closePort()
