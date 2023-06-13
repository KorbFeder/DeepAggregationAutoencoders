from utils.metrics import Metrics
import logging

def print_avg_loss(train_metrics: Metrics, test_metrics: Metrics, name: str):
	logging.info('-------------------------------------')
	logging.info(f'Average last episode training loss for {name}')
	logging.info(f'\t{[loss[-1] for loss in train_metrics.episodic_loss]}')
	logging.info('-------------------------------------')
	logging.info(f'Average testing loss for {name}')
	test_metrics.print_avg_loss()
	logging.info('-------------------------------------')
	logging.info(f'Training time: {train_metrics.curr_time_elapsed} for {name}')
	logging.info(f'Testing time: {test_metrics.curr_time_elapsed - train_metrics.curr_time_elapsed} for {name}')
	logging.info('-------------------------------------')


