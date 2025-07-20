import sys
import logging

# Exception Handler Function
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in Python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))
    return error_message

# Custom Exception Class
class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        self.error_message = error_message_detail(error, error_detail)
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message

# Main Test Block
if __name__ == "__main__":
    logging.basicConfig(filename="error.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by zero exception occurred")
        raise CustomException(e, sys)

        