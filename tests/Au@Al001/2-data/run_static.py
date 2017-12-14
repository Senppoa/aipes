#! /usr/bin/env python
"""Program for potential energy and forces calculation."""

from ase.io import read, write
from ase.calculators.emt import EMT
from ase.neb import NEB


def main():
    # --------------------------------------------------------------------------
    # Declare controlling parameters
    initial_file = "initial.traj"
    final_file = "final.traj"
    num_inter_images = 50

    # --------------------------------------------------------------------------
    # Load initial and final images
    initial_image = read(initial_file, index=-1)
    final_image = read(final_file, index=-1)

    # Interpolate
    images = interpolate(initial_image, final_image, num_inter_images)

    # Calculate potential energy and forces for each image in images
    for image in images:
        print("Dealing with image # %d." % images.index(image), flush=True)
        image.set_calculator(gen_calc())
        image.get_potential_energy(apply_constraint=False)
        image.get_forces(apply_constraint=False)
        write("static.traj", images)


def gen_calc():
    """Generate an EMT calculator."""
    return EMT()


def interpolate(initial_image, final_image, num_inter_images):
    """Interpolate N intermediate images between initial and final images."""
    images = [initial_image]
    for i in range(num_inter_images):
        images.append(initial_image.copy())
    images.append(final_image)
    neb = NEB(images)
    neb.interpolate("idpp")
    return images


if __name__ == "__main__":
    main()
