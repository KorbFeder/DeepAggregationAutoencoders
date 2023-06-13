import logging

def configure_logger(level: int, file_save_path: str):
	handlers = [logging.FileHandler(file_save_path), logging.StreamHandler()]
	logging.basicConfig(level=level, handlers=handlers)