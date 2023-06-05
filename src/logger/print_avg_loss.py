from utils.metrics import Metrics

def print_avg_loss(train_metrics: Metrics, test_metrics: Metrics, name: str):
	print('-------------------------------------')
	print(f'Average last episode training loss for {name}')
	print(f'\t{[loss[-1] for loss in train_metrics.episodic_loss]}')
	print('-------------------------------------')
	print(f'Average testing loss for {name}')
	test_metrics.print_avg_loss()
	print('-------------------------------------')
	print(f'Training time: {train_metrics.curr_time_elapsed} for {name}')
	print(f'Testing time: {test_metrics.curr_time_elapsed - train_metrics.curr_time_elapsed} for {name}')
	print('-------------------------------------')


