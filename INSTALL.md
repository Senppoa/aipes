After the installation of ASE, add the following content to ase/parallel.py

    try:
        import mpi4py
    except ImportError:
        pass

