from dataclasses import dataclass
import logging
folder = "football_games"
import os
import re
import zipfile


@dataclass
class MatchFile:
    datafile: str
    metafile: str
    annotatedfile: str = None
    name: str = ""
    home: str = ""
    away: str = ""
    index: int = None

def find_files(folder: str, decompress: bool=False):
    # Decompress positional data
    if decompress:
        for root, dirs, files in os.walk(folder):
            for f in files:
                    file_path = os.path.join(root, f)
                    if f.endswith('.zip'):
                        # Extract ZIP files
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(os.path.join(root, f.split('.')[0]))
    # print(folder)
    matches = []
    for root, dirs, files in os.walk(folder):
        # print(root, dirs, files)
        if len(dirs) == 0:
            # detect .dat and .xml files
            dat_file = None
            xml_file = None
            ann_file = None
            for f in files:
                pattern_dats = r'(\d+.dat|PosData_Live.dat|[A-Z]{3}-[A-Z]{3}.dat)'
                pattern_xml = r'(\d+_metadata.xml|PosData_Live.xml|[A-Z]{3}-[A-Z]{3}_metadata.xml)'
                pattern_ann = r".*annotation\.npz$"
                if re.search(pattern_dats, f):
                    dat_file = f
                elif re.search(pattern_xml, f):
                    xml_file = f
                elif re.search(pattern_ann, f):
                    ann_file = f


            if dat_file is None or xml_file is None or ann_file is None:
                # if dat_file is None:
                #     logging.warning(f"{root} does not contain dat file")
                # if xml_file is None:
                #     logging.warning(f"{root} does not contain xml file")
                continue
            
            # Get info fo MatchFile class 
            pattern = r'\b[A-Z]{3}-[A-Z]{3}'
            teams = re.findall(pattern, root)[0]
            
            home, away = teams.split("-")
            

            
            # if ann_file is not None:
            #     ann_file = f"{root}/{ann_file}"

            # Check
            if (dat_file[:-4] == xml_file[:-13]) and (ann_file is not None):
                m = MatchFile(f"{root}/{dat_file}", f"{root}/{xml_file}", f"{root}/{ann_file}", teams, home, away)
                matches.append(m)

            elif dat_file[:-4] == xml_file[:-4] and (ann_file is not None):
                m = MatchFile(f"{root}/{dat_file}", f"{root}/{xml_file}", f"{root}/{ann_file}", teams, home, away)
                matches.append(m)
            
            elif (re.match(r'\d+_metadata', xml_file)) and ("PosData_Live" in dat_file) and (ann_file is not None):
                m = MatchFile(f"{root}/{dat_file}", f"{root}/{xml_file}", f"{root}/{ann_file}", teams, home, away)
                matches.append(m)

    return matches

# matches = find_files(folder)
