# -*- coding: utf-8 -*-

"""
@article {Chaillet2026.04.29.721716,
	author = {Chaillet, Marten L. and van Loenhout, Joyce and Leung, Miguel R. and Burt, Alister and Tegunov, Dimitry},
	title = {MissAlignment Teaches Itself Better Cryo-ET Tilt-Series Alignment by Making It Worse},
	elocation-id = {2026.04.29.721716},
	year = {2026},
	doi = {10.64898/2026.04.29.721716},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Cryo-electron tomography (cryo-ET) visualizes cellular environments in a near-native state at macromolecular resolution. Accurate alignment of the tilted projection images is essential for data interpretation, yet existing reference-free algorithms often fail due to limited information overlap between images and inaccurate assumptions about the sample. Human experts, however, can easily recognize the misalignment left by these tools. We introduce MissAlignment, a machine learning approach that trains similar intuition to improve tilt-series alignment. A convolutional neural network learns to score alignment accuracy using a contrastive loss that does not require well-aligned ground truth, and gradient back-propagation from this score optimizes individual image alignment parameters. MissAlignment significantly outperforms reference-free techniques, rivals reference-based alignment, and improves all downstream analyses, making cryo-ET applicable to a broader range of biological samples.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2026/05/02/2026.04.29.721716},
	eprint = {https://www.biorxiv.org/content/early/2026/05/02/2026.04.29.721716.full.pdf},
	journal = {bioRxiv}
}
"""