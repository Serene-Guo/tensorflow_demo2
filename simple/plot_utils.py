import matplotlib.pyplot as plt
import numpy as np


###  for testing
def plot_image(original_images, noisy_images, reconstructed_images):
	"""
	Create figure of original and reconstruced image.
	param original_image: original images to be plotted, (?, img_h * img_w)
	param noisy_image: original images to be plotted, (?, img_h * img_w)
	param reconstruced_image: reconstructed images to be plotted, (?, img_h*img_w)
	"""
	num_images = original_images.shape[0]
	fig, axes = plt.subplots(num_images, 3, figsize=(9, 9))
	fig.subplots_adjust(hspace=.1, wspace=0)

	img_h = img_w = np.sqrt(original_images.shape[-1]).astype(int)
	
	for i, ax in enumerate(axes):
		# Plot image
		ax[0].imshow(original_images[i].reshape((img_h, img_w)), cmap='gray')
		ax[1].imshow(noisy_images[i].reshape((img_h, img_w)), cmap='gray')
		ax[2].imshow(reconstructed_images[i].reshape((img_h, img_w)), cmap='gray')
	
		# Remove ticks from the plot
		for sub_ax in ax:
			sub_ax.set_xticks([])
			sub_ax.set_yticks([])
	
	for ax, col in zip(axes[0], ["Original Image", "Noisy Image", "Reconstructed Image"]):
		ax.set_title(col)

	fig.tight_layout()
	plt.show()
