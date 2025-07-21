import sys
import logging  # use Python's logging or import from your custom logger if available

# Exception Handler Function
def error_message_detail(error, error_detail: sys):
    exc_type, exc_value, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in Python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))
    return error_message

# Custom Exception Class
class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(error_message_detail(error, error_detail))
        self.error_message = error_message_detail(error, error_detail)


    def __str__(self):
        return self.error_message

