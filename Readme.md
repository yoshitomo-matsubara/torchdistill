torchdistill: un marco modular basado en la configuración para la destilación del conocimiento
Versión de PyPI Estado de construcción

torchdistill (anteriormente kdkit ) ofrece varios métodos de destilación de conocimientos y le permite diseñar (nuevos) experimentos simplemente editando un archivo yaml en lugar de código Python. Incluso cuando necesite extraer representaciones intermedias en modelos profesor / alumno, NO necesitará volver a implementar los modelos, que a menudo cambian la interfaz del reenvío, sino que especifiquen la (s) ruta (s) del módulo en el archivo yaml.

Administrador de gancho hacia adelante
Con ForwardHookManager , puede extraer representaciones intermedias en el modelo sin modificar la interfaz de su función de avance.
Este cuaderno de ejemplo le dará una mejor idea del uso.

Precisión de validación de primer nivel para ILSVRC 2012 (ImageNet)
T: ResNet-34 *	Preentrenado	KD	A	PIE	CRD	Tf-KD	SSKD	L2	PAD-L2
S: ResNet-18	69,76 *	71,37	70,90	71,56	70,93	70,52	70.09	71.08	71,71
Trabajo original	N / A	N / A	70,70	71,43 **	71,17	70,42	71,62	70,90	71,71
* Torchvision proporciona los modelos ResNet-34 y ResNet-18 previamente entrenados.
** FT se evalúa con ILSVRC 2015 en el trabajo original.
Para la segunda fila (S: ResNet-18), el punto de control (pesos entrenados), los archivos de configuración y de registro están disponibles , y las configuraciones reutilizan los hiperparámetros, como el número de épocas utilizadas en el trabajo original, excepto KD.

Ejemplos
El código ejecutable se puede encontrar en ejemplos / como

Clasificación de imágenes : ImageNet (ILSVRC 2012), CIFAR-10, CIFAR-100, etc.
Detección de objetos : COCO 2017, etc.
Segmentación semántica : COCO 2017, PASCAL VOC, etc
Ejemplos de Google Colab
CIFAR-10 y CIFAR-100
Formación sin modelos docentes Abrir en Colab
Destilación del conocimiento Abrir en Colab
Estos ejemplos están disponibles en demo / . Tenga en cuenta que los ejemplos son para usuarios de Google Colab y, por lo general, los ejemplos / serían una mejor referencia si tiene sus propias GPU.

Citación
[ Preimpresión ]

@article { matsubara2020torchdistill ,
   title = { torchdistill: A Modular, Configuration-Driven Framework for Knowledge Distillation } ,
   author = { Matsubara, Yoshitomo } ,
   year = { 2020 } 
  eprint = { 2011.12913 } ,
   archivePrefix = { arXiv } ,
   primaryClass = { cs.LG } 
}
Como instalar
Python 3.6> =
pipenv (opcional)
Instalar por pip / pipenv
pip3 install torchdistill
# or use pipenv
pipenv install torchdistill
Instalar desde este repositorio
git clone https://github.com/yoshitomo-matsubara/torchdistill.git
cd torchdistill/
pip3 install -e .
# or use pipenv
pipenv install "-e ."
Problemas / Contacto
La documentación está en proceso. Mientras tanto, siéntase libre de crear un problema si tiene una solicitud de función o envíeme un correo electrónico ( yoshitom@uci.edu ) si desea preguntarme en privado.

Referencias
🔍 pytorch / visión / referencias / clasificación /
🔍 pytorch / visión / referencias / detección /
🔍 pytorch / vision / referencias / segmentación /
🔍Geoffrey Hinton, Oriol Vinyals y Jeff Dean. "Destilar el conocimiento en una red neuronal" (Taller de aprendizaje profundo y aprendizaje de representación: NeurIPS 2014)
🔍Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta y Yoshua Bengio. "FitNets: sugerencias para redes finas y profundas" (ICLR 2015)
🔍Junho Yim, Donggyu Joo, Jihoon Bae y Junmo Kim. "Un regalo de la destilación del conocimiento: optimización rápida, minimización de la red y aprendizaje de transferencia" (CVPR 2017)
🔍Sergey Zagoruyko y Nikos Komodakis. "Prestar más atención a la atención: mejorar el rendimiento de las redes neuronales convolucionales mediante la transferencia de atención" (ICLR 2017)
🔍Nikolaos Passalis y Anastasios Tefas. "Aprendizaje de representaciones profundas con transferencia probabilística de conocimiento" (ECCV 2018)
🔍Jangho Kim, Seonguk Park y Nojun Kwak. "Red compleja parafraseada: compresión de red mediante transferencia de factores" (NeurIPS 2018)
🔍Byeongho Heo, Minsik Lee, Sangdoo Yun y Jin Young Choi. "Transferencia de conocimiento a través de la destilación de los límites de activación formados por neuronas ocultas" (AAAI 2019)
🔍Wonpyo Park, Dongju Kim, Yan Lu y Minsu Cho. "Destilación del conocimiento relacional" (CVPR 2019)
🔍Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D. Lawrence y Zhenwen Dai. "Destilación de información variacional para la transferencia de conocimientos" (CVPR 2019)
🔍Yoshitomo Matsubara, Sabur Baidya, Davide Callegaro, Marco Levorato y Sameer Singh. "Redes neuronales profundas divididas destiladas para sistemas en tiempo real asistidos por bordes" (Taller sobre temas candentes en análisis de video y bordes inteligentes: MobiCom 2019)
🔍Baoyun Peng, Xiao Jin, Jiaheng Liu, Dongsheng Li, Yichao Wu, Yu Liu, Shunfeng Zhou y Zhaoning Zhang. "Congruencia de correlación para la destilación del conocimiento" (ICCV 2019)
🔍Frederick Tung y Greg Mori. "Destilación del conocimiento que preserva la similitud" (ICCV 2019)
🔍Yonglong Tian, ​​Dilip Krishnan y Phillip Isola. "Destilación de representación contrastiva" (ICLR 2020)
🔍Yoshitomo Matsubara y Marco Levorato. "Compresión y filtrado neuronales para la detección de objetos en tiempo real asistida por bordes en redes desafiadas" (ICPR 2020)
🔍Li Yuan, Francis EHTay, Guilin Li, Tao Wang y Jiashi Feng. "Revisando la destilación de conocimientos mediante la regularización de suavizado de etiquetas" (CVPR 2020)
🔍Guodong Xu, Ziwei Liu, Xiaoxiao Li y Chen cambian a Loy. "La destilación del conocimiento se encuentra con la auto-supervisión" (ECCV 2020)
🔍Youcai Zhang, Zhonghao Lan, Yuchen Dai, Fangao Zeng, Yan Bai, Jie Chang y Yichen Wei. "Destilación adaptativa Prime-Aware" (ECCV 2020)
