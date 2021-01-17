def load():

    # Ugly hack to allow absolute import from the root folder
    # whatever its name is. Please forgive the heresy.
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    return "tbj_2021"