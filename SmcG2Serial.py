# Uses the pySerial library to send and receive data from a
# Simple Motor Controller G2.
#
# NOTE: The Simple Motor Controller's input mode must be "Serial/USB".
# NOTE: You might need to change the "port_name =" line below to specify the
#   right serial port.
 
import serial
 
class SmcG2Serial(object):
  def __init__(self, port, device_number=None):
    self.port = port
    self.device_number = device_number
 
  def send_command(self, cmd, *data_bytes):
    if self.device_number == None:
      header = [cmd]  # Compact protocol
    else:
      header = [0xAA, device_number, cmd & 0x7F]  # Pololu protocol
    self.port.write(header + list(data_bytes))
 
  # Sends the Exit Safe Start command, which is required to drive the motor.
  def exit_safe_start(self):
    self.send_command(0x83)
 
  # Sets the SMC's target speed (-3200 to 3200).
  def set_target_speed(self, speed):
    cmd = 0x85  # Motor forward
    if speed < 0:
      cmd = 0x86  # Motor reverse
      speed = -speed
    self.send_command(cmd, speed & 0x1F, speed >> 5 & 0x7F)
 
  # Gets the specified variable as an unsigned value.
  def get_variable(self, id):
    self.send_command(0xA1, id)
    result = self.port.read(2)
    if len(result) != 2:
      raise RuntimeError("Expected to read 2 bytes, got {}."
        .format(len(result)))
    b = bytearray(result)
    return b[0] + 256 * b[1]
 
  # Gets the specified variable as a signed value.
  def get_variable_signed(self, id):
    value = self.get_variable(id)
    if value >= 0x8000:
      value -= 0x10000
    return value
 
  # Gets the target speed (-3200 to 3200).
  def get_target_speed(self):
    return self.get_variable_signed(20)
 
  # Gets a number where each bit represents a different error, and the
  # bit is 1 if the error is currently active.
  # See the user's guide for definitions of the different error bits.
  def get_error_status(self):
    return self.get_variable(0)
 
# Choose the serial port name.
# Linux USB example:  "/dev/ttyACM0"  (see also: /dev/serial/by-id)
# macOS USB example:  "/dev/cu.usbmodem001234562"
# Windows example:    "COM6"
# port_name = "COM3"
 
# Choose the baud rate (bits per second).  This does not matter if you are
# connecting to the SMC over USB.  If you are connecting via the TX and RX
# lines, this should match the baud rate in the SMC's serial settings.
# baud_rate = 9600
 
# Change this to a number between 0 and 127 that matches the device number of
# your SMC if there are multiple serial devices on the line and you want to
# use the Pololu Protocol.
# device_number = None
 
# port = serial.Serial(port_name, baud_rate, timeout=0.1, write_timeout=0.1)
 
# smc = SmcG2Serial(port, device_number)
 
# smc.exit_safe_start()
 
# error_status = smc.get_error_status()
# print("Error status: 0x{:04X}".format(error_status))
 
# target_speed = smc.get_target_speed()
# print("Target speed is {}.".format(target_speed))
 
# new_speed = 3200 if target_speed <= 0 else -3200
# print("Setting target speed to {}.\n".format(new_speed));
# smc.set_target_speed(new_speed)

# import time

# time.sleep(1)
# smc.set_target_speed(0)