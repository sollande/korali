project('korali', ['c', 'cpp'],
  version: '3.0.0',
  license: 'MIT', # or also later?
  default_options: [
    'cpp_std=c++17',
    'b_ndebug=if-release',
    'b_asneeded=false',
    'default_library=shared'
    ]
)
summary({
  'buildtype': get_option('buildtype'),
  'default_library': get_option('default_library'),
  }, section: 'Build')

# compiler object
cpp = meson.get_compiler('cpp')

if get_option('buildtype').startswith('debug')
  add_project_arguments('-DDEBUG', language : 'cpp')
endif

# configuration object
korali_conf = configuration_data()

# python installation
pymodule = import('python')
python = pymodule.find_installation('python3', required: true)

korali = 'korali' # python package name
korali_install_base = python.get_install_dir() / korali
korali_install_headers = korali_install_base / 'include'
summary({
  'prefix': korali_install_base,
  'python': python.language_version()
  }, section: 'Python')

# Code coverage configuration
if get_option('b_coverage')
  gcov_args = [
    '-fno-inline',
    '-Wno-error=cpp', # bypass _FORTIFY_SOURCE
  ]
  if cpp.get_id().to_lower() == 'gcc'
    gcov_args += [
      '-fno-default-inline',
      '-fno-inline-small-functions',
    ]
  endif
  add_project_arguments(gcov_args, language: ['cpp'])
endif

# list of Korali dependencies
null_dep = dependency('', required : false) # used for simplification of code
korali_include = []
korali_deps = []

# process required korali dependencies
# TODO: [fabianw@mavt.ethz.ch; 2021-02-13] should probably test for system cblas
# (needs benchmark: gslcblas versus system cblas (and possibly ATLAS))
korali_deps += dependency('gsl', fallback: ['gsl', 'gsl_dep'], version : '>=2.5', required: true)
korali_deps += dependency('eigen3', fallback: ['eigen', 'eigen_dep'], required: true)

# Process pybind11
pybind11_dep = dependency('pybind11', fallback: ['pybind11', 'pybind11_dep'], required: true)
korali_deps += [ pybind11_dep ]

# process optional korali dependencies
mpi_dep = null_dep
openmp_dep = null_dep

# MPI - Not required because resolution of MPI dependencies can be introduced by the compiler wrapper
# For example: Cray systems embed MPI dependencies on CC
if get_option('mpi')
  mpi_dep = dependency('mpi', language: 'cpp', required: false)
    
  if mpi_dep.found() == false
   mpi_dep = dependency('mpich', required: false)
  endif
endif

mpi4py_found = false
if get_option('mpi4py')
 assert(get_option('mpi'), 'MPI4Py can only be requested if MPI is also requested.')
 mpi4py_include = run_command(python, '-c', 'import mpi4py; print(mpi4py.get_include())').stdout().strip()
 assert(mpi4py_include.contains('include'), 'Could not find a correct MPI4Py Python module support.')
 message('Using mpi4py from ' + mpi4py_include)
 mpi4py_found = true
 korali_include += [ mpi4py_include ]
endif

if get_option('openmp')
  openmp_dep = dependency('openmp', required: false)
endif
korali_deps += [mpi_dep, openmp_dep]

onednn_dep = null_dep
if get_option('onednn') or get_option('onednn_path') != ''
  if get_option('onednn_path') != '' # prioritized if given
    # this does not resolve other dependencies the library may depend on
    onednn_dep = cpp.find_library('dnnl',
      dirs: [
        get_option('onednn_path') / 'lib',
        get_option('onednn_path') / 'lib64'],
      required: true)
    inc = get_option('onednn_path') / 'include'
    korali_include += include_directories(inc)
    assert(cpp.has_header('dnnl.h', args: '-I' + inc))
  elif get_option('onednn')
    # preferred dependency resolution via pkg-config or cmake
    onednn_dep = dependency('dnnl', required: true)
  endif
endif
korali_deps += onednn_dep

cudnn_dep = null_dep
if get_option('cudnn') or get_option('cudnn_path') != ''
  # korali internal dependency
  korali_deps += dependency('cuda', required: true)

  # cuDNN does not provide any pkg-config or other config tools, assume it does
  # not depend on other libraries
  if get_option('cudnn_path') != '' # prioritized if given
    cudnn_dep = cpp.find_library('cudnn',
      dirs: [
        get_option('cudnn_path') / 'lib',
        get_option('cudnn_path') / 'lib64'],
      required: true)
    inc = get_option('cudnn_path') / 'include'
    korali_include += include_directories(inc)
    assert(cpp.has_header('cudnn.h', args: '-I' + inc))
  elif get_option('cudnn')
    # cuDNN is installed in system path but not discoverable
    cudnn_dep = cpp.find_library('cudnn', required: true)
    assert(cpp.has_header('cudnn.h'))
  endif
endif
korali_deps += cudnn_dep


# set configuration values
korali_conf.set('_KORALI_USE_MPI', get_option('mpi'),
  description: 'Use MPI for distributed memory parallelism',
)
korali_conf.set('_KORALI_USE_MPI4PY', mpi4py_found,
  description: 'Use MPI4Py for distributed memory parallelism on Python',
)
korali_conf.set('_KORALI_USE_ONEDNN', onednn_dep.found(),
  description: 'Use Intel OndeDNN backend for Deep Neural Networks',
)
korali_conf.set('_KORALI_USE_CUDNN', cudnn_dep.found(),
  description: 'Use Nvidia CUDNN backend for Deep Neural Networks (requires CUDA)',
)
summary({
  'MPI': get_option('mpi'),
  'MPI4Py': mpi4py_found,
  'OpenMP': openmp_dep.found(),
  'oneDNN': onednn_dep.found(),
  'cuDNN': cudnn_dep.found(),
  }, section: 'Dependencies')

# process korali extension
subdir('source')

# process korali pure python
subdir('python/korali')

# tests
if get_option('test')
  nomalloc = environment({'MALLOC_PERTURB_': '0'})
  subdir('tests')
  subdir('examples')
endif
