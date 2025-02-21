# PhonoQ
PhonoQ (Phono Cue) is a deep learning model used to compute phonetic-based features related to duration, rate, rhythm*, and goodness of pronunciation* of 18 phonological classes

PhonoQ converts a sequence of speech frames $\vec{S_t} = \{\vec{s}_0,\vec{s}_1,..., \vec{s}_{T-1}\}$ into a sequence of phoneme posterior probabilities $\vec{Y_t}[\vec{z}] = \{\vec{y}_0[\vec{z}],\vec{y}_1[\vec{z}],..., \vec{y}_{T-1}[\vec{z}]\}$, where $\vec{z}=1,2,\dots,z,\dots,Z$ are all the possible phoneme groups~\citep{Cernak2015phonological}; 
thus, $y_t[z]$ is the probability of occurrence of the $z$-th phoneme class in the $t$-th speech frame.
In this thesis, phoneme precision is evaluated considering three main dimensions: 
\begin{enumerate}
    \item Manner of articulation: Refers to the way the speech articulators are set so that different consonants and vowels can be produced.
    \item Place of articulation: The point of contact where an obstruction occurs in the vocal tract in order to produce different speech sounds
    \item Voicing: Activity of the vocal folds, i.e., whether a phoneme is voiced or voiceless
\end{enumerate}

#THIS PAGE IS UNDER CONSTRUCTION

# Cite as
- Arias-Vergara, T. (2022). Analysis of Pathological Speech Signals. Logos Verlag Berlin GmbH,Vol. 50.

# Related publications
 - Arias-Vergara, T., Pérez-Toro P.A., Liu X., Xing F., Stone M., Zhuo J., Prince J., Schuster M., Noeth E., Woo J., & Maier, A. (2024). Contrastive Learning Approach for Assessment of Phonological Precision in Patients with Tongue Cancer Using MRI Data. In Proceedings of the 25th  Interspeech, 927-931.
 
 - Arias-Vergara, T., Londoño-Mora, E., Pérez-Toro, P.A., Schuster, M., Nöth, E., Orozco-Arroyave, J.R., Maier, A. (2023) Measuring Phonological Precision in Children with Cleft Lip and Palate. Proceedings of the 24th INTERSPEECH, pp. 4638-4642.

- Pérez-Toro, P. A., Rodríguez-Salas, D., Arias-Vergara, T., Bayerl, S. P., Klumpp, P., Riedhammer, K., ... & Orozco-Arroyave, J. R. (2023, June). Transferring Quantified Emotion Knowledge for the Detection of Depression in Alzheimer’s Disease Using Forestnets. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1-5.

- Pérez-Toro, P.A., Arias-Vergara, T., Braun, F., Hönig, F., Tobón-Quintero, C.A., Aguillón, D., Lopera, F., Hincapié-Henao, L., Schuster, M., Riedhammer, K., Maier, A., Nöth, E., Orozco-Arroyave, J.R. (2023) Automatic Assessment of Alzheimer's across Three Languages Using Speech and Language Features. Proceedings of the 24th INTERSPEECH, pp. 1748-1752.


