.. _my-ref-to-z2pack-tutorial:

Running a Z2Pack calculation
============================

.. toctree::
   :maxdepth: 2

Standard ``Z2pack`` input script
--------------------------------

Example of a Z2pack input file using QuantumESPRESSO (taken from the `Z2pack dodcumentation <http://z2pack.ethz.ch/doc/2.1/examples/espresso.html>`_)

.. code-block:: python

   #!/usr/bin/env python
   # -*- coding: utf-8 -*-

   import os
   import shutil
   import subprocess

   import matplotlib.pyplot as plt
   import xml.etree.ElementTree as ET

   import z2pack

   # Edit the paths to your Quantum Espresso and Wannier90 here
   qedir = '/home/greschd/software/espresso-5.4.0/bin/'
   wandir = '/home/greschd/software/wannier90-1.2'

   # Commands to run pw, pw2wannier90, wannier90
   mpirun = 'mpirun -np 4 '
   pwcmd = mpirun + qedir + '/pw.x '
   pw2wancmd = mpirun + qedir + '/pw2wannier90.x '
   wancmd = wandir + '/wannier90.x'

   z2cmd = (wancmd + ' bi -pp;' +
            pwcmd + '< bi.nscf.in >& pw.log;' +
            pw2wancmd + '< bi.pw2wan.in >& pw2wan.log;')

   # creating the results folder, running the SCF calculation if needed
   if not os.path.exists('./plots'):
       os.mkdir('./plots')
   if not os.path.exists('./results'):
       os.mkdir('./results')
   if not os.path.exists('./scf'):
       os.makedirs('./scf')
       print("Running the scf calculation")
       shutil.copyfile('input/bi.scf.in', 'scf/bi.scf.in')
       out = subprocess.call(pwcmd + ' < bi.scf.in > scf.out', shell=True, cwd='./scf')
       if out != 0:
           raise RuntimeError('Error in SCF call. Inspect scf folder for details, and delete it to re-run the SCF calculation.')

   # Copying the lattice parameters from bi.save/data-file.xml into bi.win
   cell = ET.parse('scf/bi.save/data-file.xml').find('CELL').find('DIRECT_LATTICE_VECTORS')
   unit = cell[0].attrib['UNITS']
   lattice = '\n '.join([line.text.strip('\n ') for line in cell[1:]])

   with open('input/tpl_bi.win', 'r') as f:
       tpl_bi_win = f.read()
   with open('input/bi.win', 'w') as f:
       f.write(tpl_bi_win.format(unit=unit, lattice=lattice))

   # Creating the System. Note that the SCF charge file does not need to be
   # copied, but instead can be referenced in the .files file.
   # The k-points input is appended to the .in file
   input_files = ['input/' + name for name in ["bi.nscf.in", "bi.pw2wan.in", "bi.win" ]]
   system = z2pack.fp.System(
       input_files=input_files,
       kpt_fct=[z2pack.fp.kpoint.qe, z2pack.fp.kpoint.wannier90],
       kpt_path=["bi.nscf.in","bi.win"],
       command=z2cmd,
       executable='/bin/bash',
       mmn_path='bi.mmn'
   )

   # Run the WCC calculations
   result_0 = z2pack.surface.run(
       system=system,
       surface=lambda s, t: [0, s / 2, t],
       save_file='./results/res_0.json',
       load=True
   )
   result_1 = z2pack.surface.run(
       system=system,
       surface=lambda s, t: [0.5, s / 2, t],
       save_file='./results/res_1.json',
       load=True
   )

   # Combining the two plots
   fig, ax = plt.subplots(1, 2, sharey=True, figsize=(9,5))
   z2pack.plot.wcc(result_0, axis=ax[0])
   z2pack.plot.wcc(result_1, axis=ax[1])
   plt.savefig('plots/plot.pdf', bbox_inches='tight')

   print('Z2 topological invariant at kx = 0: {0}'.format(z2pack.invariant.z2(result_0)))
   print('Z2 topological invariant at kx = 0.5: {0}'.format(z2pack.invariant.z2(result_1)))


The input file here shown requires several inputs to be able to be run on any possible system:
   - the MPI command
   - the path to pw.x, pw2wannier90.x and wannier90.x codes
   - the way the codes takes input files (eg: '<' or '-in')
   - The path to already prepared QE input files for the scf, nscf and pw2wannier90 calculations
   - The path to already prepared Wannier90.x input file
   - The surface on which the HWCC have to be calculated (see `documentation <http://z2pack.ethz.ch/doc/2.1/tutorial/surface.html>`_ for more details)

This plugin, after having set the required codes for a computer within AiiDA, will take care of building the input file for you.

Running a z2pack calculation with AiiDA
---------------------------------------

Now we are going to prepare a script to submit a job to your local installation of AiiDA.

Let's say that through the ``verdi`` command you have already installed
a cluster, say ``my_cluster``, and that you also compiled Quantum ESPRESSO and Wannier90
on the cluster, and installed the codes:
   - `pw.x` with ``verdi`` with label ``pw-6.3``
   - `pw2wannier90.x` with ``verdi`` with label ``pw2wannier90-6.3``
   - `wannier90.x` with ``verdi`` with label ``wannier90-2.1.0``
   - `python3 + z2pack` with ``verdi`` with label ``z2pack_py3.8.2``

for instance, so that in the rest of this tutorial we will reference to the pw
code as ``pw-6.3@my_cluster``.

Let's start writing the python script.
First of all, we need to load the configuration concerning your
particular installation, in particular, the details of your database installation::

  #!/usr/bin/env python3
  from aiida import load_profile
  load_profile()


Code
----


Now we have to select the code. Note that in AiiDA the object 'code' in the
database is meant to represent a specific executable, i.e. a given
compiled version of a code. Every calculation in AiiDA is linked to a *code*, installed on
a specific *computer*.
This means that if you install Quantum ESPRESSO on two computers *A* and *B*,
you will need to have two different 'codes' in the database
(although the source of the code is the same, the binary file is different).

If you setup the code as previously described correctly, then it is
sufficient to write:

.. code-block:: python

   from aiida.orm.utils import load_code
   computer = 'my_cluster'

   pw_code        = load_code(f'pw-6.3@{computer}')
   overlap_code   = load_code(f'pw2wannier90-6.3@{computer}')
   wannier90_code = load_code(f'wannier90-2.1.0@{computer}')
   z2pack_code    = load_code(f'z2pack_py3.8.2@{computer}')

Preparing the inputs
--------------------

The z2pack calculation plugin provided can work from 2 different starting points:
   - Start from a previously run ``scf`` calculation.
   - Re-Start from a previously run ``z2pack`` calculation.

The inputs can be passed as a dictionary when launching the calculation:

.. code-block:: python

   from aiida.engine import submit
   from aiida.plugins import CalculationFactory

   Z2packCalc == CalculationFactory('z2pack.z2pack')

   inputs = {
     'parent_folder':load_node(...),
     # ...
   }

   res = submit(Z2packCalc, **inputs)

or through the builder:

.. code-block:: python

   from aiida.engine import submit
   from aiida.plugins import CalculationFactory

   Z2packCalc == CalculationFactory('z2pack.z2pack')

   builder = Z2packCalc.get_builder()

   builder.code = ...
   builder.pw_code = ...
   # ...
   builder.metadata. ... = ...

   res = submit(builder)

The builder is easier to use from the verdi shell, as TAB autocompletion is enabled, and using ``?`` (eg: `builder.parameter_name?`) will give information on every parameter.

List of possible inputs
+++++++++++++++++++++++
   - `parent_folder` (orm.RemoteData): Output remote node of the ``scf`` calculation
   - `pw_parameters` (orm.Dict): parameters for the ``nscf`` calculations (see `aiida-quantumespresso documentation <https://aiida-quantumespresso.readthedocs.io/en/latest/user_guide/get_started/examples/pw_tutorial.html#parameters>`_ for more info)
   - `pw_settings` (orm.Dict): Optional settings for the pw calculations (see `aiida-quantumespresso documentation <https://aiida-quantumespresso.readthedocs.io/en/latest/user_guide/get_started/examples/pw_tutorial.html#preparing-a-calculation>`_)
   - `overlap_parameters` (orm.Dict): Optional settings for the pw2wannier90.x input. The `key`:`value` pair represent a member of the namelist in the input file and its respective value.
   - `wannier90_parameters` (orm.Dict): Optional settings for the wannier90.x inputs. Here is an example for a material with 28 filled bands where the calculated conduction bands (from 29 to 35) needs to be ignored:

   .. code-block:: python

      wannier90_parameters = {
        'num_wann':28,
        'num_bands':28,
        'exclude_bands':[*range(29,35)],
        }

   - `z2pack_settings` (orm.Dict): settings for the Z2pack calculation:

      - Required key/value pairs:
         - `dim_mode` (string): Possible values '2D' or '3D'. Specify the dimensionality of the system under study. For 2D system, the surface is automatically set to ``lambda t1,t2: [t2, t1/2, 0]`` for Z2 calculations and ``lambda t1,t2: [t1, t2, 0]`` for Chern calculations. For '3D', the surface must be specified manually.
         - `invariant` (String): Invariant to be calculated. Can be either 'Z2' or 'Chern'.
         - `surface` (String): Only required for '3D' calculations. Must be a string of a python lambda function accepting 2 parameters (eg: ``lambda t1,t2: [t1, t2, 0]``)

      - Optional key/value pairs:
         - `parent_folder_symlink` (bool, default=False): If True, a symlink to the parent RemoteData folder is created instead of a copy.
         - `restart_mode` (bool, default=True): If False, restarting from a previous ``Z2pack`` calculation will only serve to inherit the input nodes and the calculation will restart from scratch.
         - `npools` (int): If specified, pools will be used (with number equal to npools) when running ``nscf`` calculations during the execution of ``Z2pack``.
         - `mpi_command` (string, default=Generated from Computer node settings): If specified, overrides the computer settings and pass a custom mpi_command.
         - `pw_in_command` (string, default='<'): How the input file is passed to QE. By default the stdin redirection via '<' is employed. Useful for some version of QE (eg: qe-gpu that only accept inputs via '-inp')
         - `pos_tol` (float): see `Z2pack documentation <http://z2pack.ethz.ch/doc/2.1/tutorial/surface.html#convergence-options>`_
         - `gap_tol` (float): see `Z2pack documentation <http://z2pack.ethz.ch/doc/2.1/tutorial/surface.html#convergence-options>`_
         - `move_tol` (float): see `Z2pack documentation <http://z2pack.ethz.ch/doc/2.1/tutorial/surface.html#convergence-options>`_
         - `num_lines` (int): see `Z2pack documentation <http://z2pack.ethz.ch/doc/2.1/tutorial/surface.html#convergence-options>`_
         - `min_neighbour_dist` (float): see `Z2pack documentation <http://z2pack.ethz.ch/doc/2.1/tutorial/surface.html#convergence-options>`_
         - `iterator` (string): string that can be evaluated as a python iterator (eg: ``range(8, 41, 2)``). See `Z2pack documentation <http://z2pack.ethz.ch/doc/2.1/tutorial/surface.html#convergence-options>`_
         - `prepend_code` (string): Code to prepend at the beginning of the python script

   - `pw_code` (orm.AbstractCode): Node for pw.x
   - `overlap_code` (orm.AbstractCode): Node for pw2wannier90.x
   - `wannier90_code` (orm.AbstractCode): Node for wannier90.x
   - `code` (orm.AbstractCode) : Ndoe for python interpreter with Z2pack installed
   - `metadata` (dict): metadata to be stored and to specify resources for a the calculation. See `AiiDA documentation <https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/processes/usage.html?highlight=metadata#process-metadata>`_ for more details.

Starting from a ``scf`` calculation
-----------------------------------

* Required inupts:
   - `parent_folder`
   - `pw_parameters`
   - `z2pack_settings`
   - `pw_code`
   - `overlap_code`
   - `wannier90_code`
   - `code`

* Optional inputs:
   - `pw_settings`
   - `overlap_parameters`
   - `wannier90_parameters`


Example of an input dictionary:

.. code-block:: python

   from aiida.orm.utils import load_node
   parent = load_node(PARENT_NODE_ID_NUMBER) # from scf calculation

   z2pack_settings = {
       'mpi_command':'mpirun -n 8',
       'dimension_mode':'2D',
       'invariant':'Z2',
       'iterator':'range(8, 81, 2)',
       'npools':4,
       'min_neighbour_dist':0.0001
       }

   inputs = {
        'parent_folder':parent,
        'pw_code':pw_code,
        'overlap_code':overlap_code,
        'wannier90_code':wannier90_code,
        'code':z2pack_code,
        'z2pack_settings':Dict(dict=z2pack_settings),
        'metadata':{
           'options':{
              'resources':{'num_machines': 1},
              'max_wallclock_seconds':1800
              }
           }
      }

Re-starting from a ``z2pack`` calculation
-----------------------------------------

* Required inupts:
   - `parent_folder`


* Optional inputs:
   - `pw_parameters`
   - `z2pack_settings`
   - `pw_code`
   - `overlap_code`
   - `wannier90_code`
   - `code`
   - `pw_settings`
   - `overlap_parameters`
   - `wannier90_parameters`

Example of an input dictionary to be used:

.. code-block:: python

   from aiida.orm.utils import load_node
   parent = load_node(PARENT_NODE_ID_NUMBER) # from z2pack calculation

   inputs = {
        'parent_folder':parent,
      }


In the case of a restart calculation, the only required parameter is the RemoteData from the previous calculation.
All other parameters are imported from the previous calculation.

If optional parameters are specified they will overwrite the respective input node from the parent.

Example of an input file
------------------------
Below is reported am example to run a Z2 calculation using the Z2pack calcjob on a monolayer of germanene.
The example assumes that the Computer and Code node have already been properly setup.
Also the RemoteData node as an output of an ``scf`` calculation should already be present in the database.

.. code-block:: python

  from aiida import orm
  from aiida import load_profile
  from aiida.orm.utils import load_node, load_code
  from aiida.plugins import CalculationFactory
  from aiida.engine import submit

  load_profile()

  Z2Calc = CalculationFactory('z2pack.z2pack')

  parent = load_node(SCF_REMOTE_DATA_ID)

  # Adjust depending on your system
  wannier90_parameters = {
      'num_wann':28, 
      'num_bands':28,
      'exclude_bands':[*range(29,35)],
      }

  z2pack_settings = {
      'mpi_command':'mpirun -n 8',
      'dimension_mode':'2D',
      'invariant':'Z2',
      'npools':4,
      'min_neighbour_dist':0.0001
      }

  computer = 'my_cluster'

  pw_code        = load_code(f'pw-6.3@{computer}')
  overlap_code   = load_code(f'pw2wannier90-6.3@{computer}')
  wannier90_code = load_code(f'wannier90-2.1.0@{computer}')
  z2pack_code    = load_code(f'z2pack_py3.8.2@{computer}')

  builder = Z2Calc.get_builder()

  builder.code = z2pack_code
  builder.pw_code = pw_code
  builder.overlap_code = overlap_code
  builder.wannier90_code = wannier90_code

  builder.wannier90_parameters = orm.Dict(dict=wannier90_parameters)
  builder.z2pack_settings = orm.Dict(dict=z2pack_settings)

  builder.metadata.options.resources = {'num_machines': 1}

  builder.parent_folder = parent

  res = submit(builder)

Use the Z2pack base workchain
=============================

The Z2pack plugin provides several workchains used to further automate the process of input file generation and error handling.

The ``Z2packBaseWorkChain`` workchain wraps the ``Z2packCalculation`` CalcJob in order to provide the following functionalities:

* Possibility to start from scratch and perform the ``scf`` calculation
* Guess wannier90_parameters from the output of the ``scf`` calculation (if not passed explicitly).
* Error handling: Restart the calculation, if it did not fail in an unrecoverable manner.

  * When exceeding the WALLTIME set in the metadata.
  * When a sanity check fails (restart with increased convergence parameters).

Base workchain inputs
+++++++++++++++++++++

* `scf` namelist: see `PwBaseWorkChain documentation <>`_ for more details. Excluded parameters:

  * `pw.structure`
  * `pw.code`
  * `clean_workdir`

* `z2pack` namelist: see :ref:`List of possible inputs`. Excluded parameters:

  * `parent_folder`
  * `pw_code`

* `structure` (orm.StructureData): Node containing the material structure
* `pw_code` (orm.AbstractCode): Node for ``pw.x``
* `clean_workdir` (orm.Bool): If True, clean the working directories of the ``pw.x`` calculations.
* `parent_folder` (orm.RemoteData): Output remote node of the ``scf`` calculation.
* `min_neighbour_distance_scale_factor` (orm.Float): If the convergence is not achieved using the `min_neighbour_distance` passed through `z2pack_settings`, restart the calculation by decreasing the value (`new_min_neighbour_distance` = `old_min_neighbour_distance` / `min_neighbour_distance_scale_factor`)
* `min_neighbour_distance_threshold_minimum` (orm.Float): Stop the iterations if `min_neighbour_distance` drops below `min_neighbour_distance_threshold_minimum`
* Other inputs in the BaseRestartWorkChain documentation (from aiida-quantumespresso v3.0.0)

If `parent_folder` is specified, the `scf` namelist will be ignored (a warning will be issued in the log if both are present).

Example of an input file
------------------------

Below is reported am example to run a Z2 calculation using the Base workchain on a monolayer of germanene.
The example assumes that the Computer and Code node have already been properly setup.
A family of full-relativistic pseudopotentials should also be loaded (eg: The PBE norm-conserving full-relativistic pseudo of 'ONCVPSP' v0.4 taken from `PseudoDojo <http://www.pseudo-dojo.org/pseudos/nc-fr-04_pbe_standard_upf.tgz>`_.

.. code-block:: python

  import numpy as np
  from aiida import load_profile

  from aiida.orm import Dict, StructureData, KpointsData
  from aiida.orm.utils import load_code
  from aiida.orm.nodes.data.upf import get_pseudos_from_structure
  from aiida.engine import submit
  from aiida.plugin import WorkflowFactory

  load_profile()

  Z2packBaseWorkChain = WorkflowFactory('z2pack.base')

  alat   = 7.643 # bohr
  vacuum = 30.0
  cell = np.array([
      [alat,    0.,                0.,],
      [-alat/2, alat*np.sqrt(3)/2, 0.,],
      [0.,      0.,                vacuum,],
      ]) * 0.52917721

  positions_cryst = np.array([
      [0.333333333, 0.666666667, -0.02139],
      [0.666666667, 0.333333333,  0.02139],
      ])
  positions_cart  = np.dot(positions_cryst, cell)

  s = StructureData(cell=cell)
  for p in positions_cart:
      s.append_atom(position=p, symbols='Ge')

  parameters = {
      'CONTROL': {
          'calculation': 'scf',
          'wf_collect': True,
      },
      'SYSTEM': {
          'ecutwfc':40,
          'nbnd':30,
          'occupations':'smearing',
          'smearing':'fermi-dirac',
          'degauss':0.00036749326,
          'noncolin':True,
          'lspinorb':True,
      },
      'ELECTRONS': {
          'conv_thr': 1.e-6,
      }
  }

  z2pack_settings = {
      'mpi_command':'mpirun -n 8',
      'dimension_mode':'2D',
      'invariant':'Z2',
      'iterator':'range(8, 81, 2)',
      'npools':4,
      # 'surface':'lambda t1, t2: [ t2, t1/2, 0.0]'
      'min_neighbour_dist':0.0001
      }

  structure = s
  pseudos   = get_pseudos_from_structure(structure, "ONCVPSP-FR")
  kpoints   = KpointsData()
  kpoints.set_kpoints_mesh([12,12,1])

  pw_code        = load_code('qe-6.3_pw')
  overlap_code   = load_code('qe-6.3_pw2wannier90')
  wannier90_code = load_code('wannier90-2.1.0')
  z2pack_code    = load_code('z2pack')

  inputs = {
    'structure':structure,
    'scf':{
      'kpoints':kpoints,
      'pw':{
        'code':pw_code,
        'pseudos':pseudos,
        'parameters':Dict(dict=parameters),
        'metadata':{
          'options':{
            'resources':{'num_machines': 1},
            'max_wallclock_seconds':1800
            }
          }
        },
      },
    'z2pack':{
      'overlap_code':overlap_code,
      'wannier90_code':wannier90_code,
      'code':z2pack_code,
      'z2pack_settings':Dict(dict=z2pack_settings),
      'metadata':{
        'options':{
          'resources':{'num_machines': 1},
          'max_wallclock_seconds':1800
          }
        }
      },
    }

  res = submit(Z2packBaseWorkChain, **inputs)
