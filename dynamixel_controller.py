from dynamixel_sdk import *

class DynamixelController:
    """
    A class to control Dynamixel motors using the Dynamixel SDK.

    Attributes:
        device_name (str): The device name (e.g., COM port on Windows or tty port on Linux).
        baudrate (int): The communication baudrate.
        protocol_version (float): The protocol version of Dynamixel motors.
    """
    
    def __init__(self, device_name, baudrate, protocol_version):
        """
        Initializes the DynamixelController class.
        
        Args:
            device_name (str): The name of the device port.
            baudrate (int): The baudrate for communication.
            protocol_version (float): The version of the communication protocol.
        """
        self.device_name = device_name
        self.baudrate = baudrate
        self.protocol_version = protocol_version
        
        # Initialize PortHandler and PacketHandler
        self.port_handler = PortHandler(self.device_name)
        self.packet_handler = PacketHandler(self.protocol_version)

        self.open_port()
        self.set_baudrate()

    def open_port(self):
        """
        Opens the communication port.
        """
        if not self.port_handler.openPort():
            raise Exception("Failed to open the port")

    def set_baudrate(self):
        """
        Sets the baudrate for the communication port.
        """
        if not self.port_handler.setBaudRate(self.baudrate):
            raise Exception("Failed to change the baudrate")

    def WRITE(self, dxl_id, command_type, command_value):
        """
        Writes a value for a specified type of command to a specific motor ID.
        
        Args:
            dxl_id (int): The ID of the Dynamixel motor.
            command_type (tuple): A tuple where the first element is the address of the command,
                                and the second element is the byte length (1, 2, or 4).
            command_value (int): The value to be written to the specified command.
        Returns:
            bool: True if the write was successful, False if there was an error.
        """
        # Unpack command type and attempt TxRx write
        address, length = command_type
        if length == 1:
            dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, address, command_value)
        elif length == 2:
            dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(self.port_handler, dxl_id, address, command_value)
        elif length == 4:
            dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(self.port_handler, dxl_id, address, command_value)
        else:
            print(f"Invalid byte length: {length}")
            return False
        # Check for communication result and error
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Communication error on motor {dxl_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
            return False
        elif dxl_error != 0:
            print(f"Packet error on motor {dxl_id}: {self.packet_handler.getRxPacketError(dxl_error)}")
            return False
        # Successful write
        return True

    def READ(self, dxl_id, command_type):
        """
        Reads a value for a specified type of command to a specific motor ID.
        
        Args:
            dxl_id (int): The ID of the Dynamixel motor.
            command_type (tuple): A tuple where the first element is the address of the command,
                                and the second element is the byte length (1, 2, or 4).
        Returns:
            int: True if the write was successful, False if there was an error.
        """
        # Unpack command type and attempt TxRx write
        address, length = command_type
        if length == 1:
            dxl_value, dxl_comm_result, dxl_error = self.packet_handler.read1ByteTxRx(self.port_handler, dxl_id, address)
        elif length == 2:
            dxl_value, dxl_comm_result, dxl_error = self.packet_handler.read2ByteTxRx(self.port_handler, dxl_id, address)
        elif length == 4:
            dxl_value, dxl_comm_result, dxl_error = self.packet_handler.read4ByteTxRx(self.port_handler, dxl_id, address)
        else:
            print(f"Invalid byte length: {length}")
            return False
        # Check for communication result and error
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Communication error on motor {dxl_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
            return False
        elif dxl_error != 0:
            print(f"Packet error on motor {dxl_id}: {self.packet_handler.getRxPacketError(dxl_error)}")
            return False
        # Successful write
        return dxl_value

    def close_port(self):
        """
        Closes the communication port.
        """
        self.port_handler.closePort()

# Example usage
# if __name__ == "__main__":
#     controller = DynamixelController('/dev/ttyUSB0', 57600, 2.0)
#     try:
#         for dxl_id in controller.dynamixel_ids:
#             controller.enable_torque(dxl_id)
#             # Additional logic to set goal positions, read positions, etc.
#     except Exception as e:
#         print(e)
#     finally:
#         controller.close_port()
