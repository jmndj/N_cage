'''
批量读取cif文件并且替换原子,晶格缩放,转换为POSCAR文件

'''



import os
import shutil

from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
from pymatgen.core.lattice import Lattice
from pymatgen.core import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar

def replace_and_delete_atom_in_structure(directory_path): 
    
    file_storage =[]

    for root, dirs, files in os.walk(directory_path):
        for dir in dirs:
            file = os.path.join(root,dir)
            for file_root, file_dirs, file_files, in os.walk(file):
                for dirfile in file_files:
                    if '.cif' in dirfile:
                        cif_file_path = os.path.join(file,dirfile)
                        file_storage.append(cif_file_path)
                        
    for cif in file_storage:
        # 读取 CIF 文件
        parser = CifParser(cif)
        structure = parser.get_structures()[0]

        # 替换某个原子
        substitution_transformation = SubstitutionTransformation({"H": "N"})
        new_structure = substitution_transformation.apply_transformation(structure)

#        composition = new_structure.composition
        covalent_radius_H = Element("H").atomic_radius
        covalent_radius_N = Element("N").atomic_radius
#        for element, count in composition.items():
#            print(element)
#            if element != "N":
#                new_structure.remove_species(element)
#        new_structure.remove_species([el.symbol for el in new_structure.species if el.symbol != "N"])

#        covalent_radius_H = Element("H").data["Covalent radius"]
#        covalent_radius_N = Element("N").data["Covalent radius"]

#        covalent_radius_H = Element("H").atomic_radius
#        covalent_radius_N = Element("N").atomic_radius
        
        # 缩放晶格
        scaling_factor = covalent_radius_N / covalent_radius_H / 1.6 # 设置缩放比例
        scaled_lattice = Lattice(new_structure.lattice.matrix * scaling_factor)
        scaled_structure = Structure(scaled_lattice, new_structure.species, new_structure.frac_coords)

        # 寻找对称性
        symmetry_finder = SpacegroupAnalyzer(scaled_structure)
        symmetry_operations = symmetry_finder.get_symmetry_operations()
        symmetric_structures = symmetry_finder.get_symmetrized_structure()
#        print((symmetry.equivalent_indices))
#        with open('symmetry_operation','a+') as f:
#            print(symmetry,file = f)
#        symmetric_structures = symmetry_finder.get_symmetrized_structure()

        # 生成新的cif文件
        writer = CifWriter(symmetric_structures,symprec=0.01)
        new_file_path = cif.split('.')[0] + 'new' + '.cif'
        writer.write_file(new_file_path)
        shutil.move(new_file_path,dir_path)   

def cif_transform_primitive_poscar(directory_path):
    
    for root, dirs, files in os.walk(directory_path):
#        print(dirs)
        for file in files:
            if '.cif' in file:

                cif_file_path = os.path.join(root,file)

                parser = CifParser(cif_file_path)

                structure = parser.get_structures()[0]
                primitive_structure = structure.get_primitive_structure(tolerance=0.01)
                
#                el = Element("Li")
                primitive_structure.remove_species([el.symbol for el in primitive_structure.species if el.symbol == "Li"])
#                primitive_structure.remove_species(el.symbol)

                cif_file_save_path = os.path.join(dir_path,file.split('.')[0]) + ".vasp"
#                print(cif_file_save_path)
                poscar = Poscar(primitive_structure)

                poscar.write_file(cif_file_save_path)
#                with open(cif_file_save_path,"w") as f:
#                    f.write("Primitive Cell\n")
#                    f.write("1.0\n")
#                    primitive_structure.to(fmt="poscar",file = f)

def cif_without_Li_primitive_poscar(directory_path):

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if 'cif' in file:

                cif_file_path = os.path.join(root,file)

                parser = CifParser(cif_file_path)

                structure = parser.get_structures()[0]
                elements = structure.composition.elements
                element_symbol = [element.symbol for element in elements]
                
#                print(type(element_symbol[0]))

                if 'Li' not in element_symbol:
                    
                    primitive_structure = structure.get_primitive_structure(tolerance=0.01)
                    cif_file_save_path = os.path.join(dir_path,file.replace('new','.').split('.')[0]) + ".vasp"
                    
                    poscar = Poscar(primitive_structure)
                    poscar.write_file(cif_file_save_path)
                    

directory_path = r"D:\N_cage\replace_atom_6"
aim_path = r"D:\N_cage"
path = "replace_atom_poscar_without_Li"
dir_path = os.path.join(aim_path,path)
os.chdir(aim_path)
os.mkdir(dir_path)
cif_without_Li_primitive_poscar(directory_path)
#cif_transform_primitive_poscar(directory_path)
#replace_and_delete_atom_in_structure(directory_path)

#for file_root, file_dirs, file_files, in os.walk(aim_path):
#    for file in file_files:
#        if '.cif' in file:
#            path = os.path.join(file,aim_path)
#            os.chdir(aim_path)
#            os.remove(file)



