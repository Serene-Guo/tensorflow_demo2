import numpy as np
import matplotlib.pyplot as plt

def plot_images(images, cls_true, cls_pred=None, title=None):
	"""
	Create figure with 3*3 sub-plots
	param images: array of images to be plotted,  (9, img_h*img_w)
	param cls_true: correspnding true labels (9, )
	param cls_pred: corresponding predict labels (9, )
	"""
	fig, axes = plt.subplots(3,3, figsize=(9, 9))
	fig.subplots_adjust(hspace=0.3, wspace=0.3)
	img_h = img_w = np.sqrt(images.shape[-1]).astype(int)
	for i, ax in enumerate(axes.flat):
		ax.imshow(images[i].reshape((img_h, img_w)), cmap='binary')
		
		if cls_pred is None:
			ax_title = "True: {}".format(cls_true[i])
		else:
			ax_title = "True: {}, Pred: {}".format(cls_true[i], cls_pred[i])

		ax.set_title(ax_title)
		
		ax.set_xticks([])
		ax.set_yticks([])

	if title:
		plt.suptitle(title, size=20)
	plt.show()


def plot_example_errors(images, cls_true, cls_pred, title=None, print_right=True):
	"""
	
	"""
	if print_right:
		incorrect = np.equal(cls_pred, cls_true)
	else:
		incorrect = np.logical_not(np.equal(cls_pred, cls_true))
	incorrect_images = images[incorrect]

	cls_pred = cls_pred[incorrect]
	cls_true = cls_true[incorrect]
	
	plot_images(images=incorrect_images[0:9],
				cls_true=cls_true[0:9],
				cls_pred=cls_pred[0:9],
				title=title)


