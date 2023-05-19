# PocketFinderGNN

PocketFinderGNN is a tool for the recognition of a critical manufacturing feature named close pocket in 3D models. The close pocket is a crucial element in the manufacturing industry, and its recognition is essential for the automation and optimization of machining processes. PocketFinderGNN converts the .stp file generated by CAD/CAM systems to a graph representation of the 3D model and utilizes a Graph Convolutional Network (GCN) to predict which nodes consist of the close pocket feature. The proposed method was implemented using PyTorch Geometric and NetworkX frameworks. We trained our model on a dataset of 576 3D models obtained from the electromechanical industry and achieved an accuracy of 95%. Our method outperforms state-of-the-art techniques and demonstrates its robustness to noise and perturbations in the input data. The results show that our approach is an effective tool for the recognition of complex manufacturing features in 3D models, which can significantly improve the efficiency and accuracy of manufacturing processes.

![PocketFinder](https://github.com/betanddontcare/PocketFinderGNN/assets/31188390/972dfb1e-1e7c-4a7f-aa34-361c8d00728e)

__NOTE:__ If you need to modify the GNN model please contact me by DM.

## Requirements & Installation

__Step 1: Install all the requirements before proceeding to next steps:__

* Python >= 3.11.3
* python3-pandas >= 2.0.0
* python3-torch >= 2.0.0
* python3-numpy >= 1.24.2
* python3-networkx >= 3.1
* python3-torch-geometric >= 2.3.0

You should install all the python3 modules using the `pip3 install *package_name*` command.

(or alternatively using: `sudo apt-get install python3-*package_name*` conmmand)

__Step 2: Prepare your .stp 3D model:__

* To prepare your model for recognition you should open .stp file by Notepad and save as .txt file later on. 
* Put this file to PocketFinderGNN folder and change the path (line 12) in main.py file.

## Running the tool

In order to successfully run the tool you need to:

* Clone (git clone) or download the project.
* Convert .stp file to .txt file using Notepad.
* Change the path by using your model.
* Run main.py file in your terminal.
* Investigate which faces make a closed pocket manufacturing feature.

## Theoretical basis

To find more details about this topic please visit:

~~LINK TO SOFTWAREX PAPER~~

## Authors
- Igor Betkier, [e-mail](mailto:igor.betkier@wat.edu.pl), [github](https://github.com/betanddontcare)
- Mateusz Oszczypała, [e-mail](mailto:mateusz.oszczypala@wat.edu.pl)
- Przemysław Betkier, [e-mail](mailto:przbetkier@gmail.com), [github](https://github.com/przbetkier)

## Contributors

We would like to extend our appreciation for their contributions to the following colleagues:

- Janusz Pobożniak
- Sergiusz Sobieski
