
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


from .main import *

from .kmeans import *

if __name__ == '__main__':
    # 'Module is being executed directly, so do stuff here'
    # https://stackoverflow.com/questions/46319694/what-does-it-mean-to-run-library-module-as-a-script-with-the-m-option
    # Can be run by doing the following in terminal
    # $ python -m ryerrabelli
    # $ python -m src.EyeTraumaAnalysis.__init__  # __init__ needs to be specified, otherwise will try to find a __main__ file
    # $ python -c "import src.EyeTraumaAnalysis"
    # $ PYTHONPATH="src" python -c "import EyeTraumaAnalysis" # Python path changed only for one line
    # $ PYTHONPATH="src" python -m EyeTraumaAnalysis   # Python path changed only for one line
    print(__version__)

