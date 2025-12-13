import os
import numpy as np

def prepare_receptor(pdb_file, ligand_code="CP6"):
    print(f"[INFO] Analizando {pdb_file} para encontrar el sitio activo ({ligand_code})...")
    
    coords = []
    clean_lines = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            # 1. Extraer coordenadas del ligando original para definir el CENTRO
            if line.startswith("HETATM") and ligand_code in line:
                try:
                    # Formato PDB estricto: X[30-38], Y[38-46], Z[46-54]
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except:
                    pass
            
            # 2. Limpiar el archivo para el receptor final
            # Nos quedamos con ATOM (proteina) y eliminamos HETATM (aguas/ligandos)
            if line.startswith("ATOM"):
                clean_lines.append(line)
            elif line.startswith("TER"):
                clean_lines.append(line)
                
    if not coords:
        print(f"[ERROR] No se encontró el ligando {ligand_code}. Revisa el código PDB.")
        return None

    # Calcular Centro Geométrico
    center = np.mean(coords, axis=0)
    print(f"[EXITO] Centro del Sitio Activo detectado: {center}")
    
    # Guardar PDB limpio
    clean_pdb = "receptor_clean.pdb"
    with open(clean_pdb, 'w') as f:
        f.writelines(clean_lines)
    print(f"[INFO] Archivo limpio guardado: {clean_pdb}")
    
    # Convertir a PDBQT usando OpenBabel (Comando de sistema)
    output_pdbqt = "receptor.pdbqt"
    print(f"[INFO] Convirtiendo a {output_pdbqt} con OpenBabel...")
    
    # Comando: obabel -ipdb clean.pdb -opdbqt -O receptor.pdbqt -xr (rigid)
    os.system(f"obabel -ipdb {clean_pdb} -opdbqt -O {output_pdbqt} -xr")
    
    if os.path.exists(output_pdbqt):
        print(f"[LISTO] Receptor preparado: {output_pdbqt}")
        return center
    else:
        print("[ERROR] Falló OpenBabel. Asegúrate de instalarlo (sudo apt-get install openbabel)")
        return None

if __name__ == "__main__":
    # Asegúrate de que tu archivo descargado se llame 'receptor.pdb'
    center = prepare_receptor("receptor.pdb", ligand_code="CP6")
    
    if center is not None:
        print("\n=== DATOS PARA EL DOCKING ===")
        print(f"center_x = {center[0]:.3f}")
        print(f"center_y = {center[1]:.3f}")
        print(f"center_z = {center[2]:.3f}")
        print("size_x = 20.0\nsize_y = 20.0\nsize_z = 20.0")