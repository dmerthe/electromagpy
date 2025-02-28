from electromagpy.fields.electrostatic import Vacuum

vacuum = Vacuum()

print(vacuum.V([0.0, 0.0, 0.0], 0.0))
