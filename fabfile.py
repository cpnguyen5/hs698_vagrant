from fabric.api import run, sudo, env
import subprocess


### LINUX PACKAGES TO INSTALL ###

INSTALL_PACKAGES = [
   'openssl',
   'libxml2-dev',
   'libxslt1-dev',
   'libssl-dev',
   'libatlas-base-dev',
   'libblas-dev',
   'libffi-dev',
   'gfortran',
   'g++',
   'python2.7-dev',
   'python-pip',
   'build-essential',
   'python-scipy',
   'pkg-config',
   'libpng-dev',
   'libfreetype6-dev',
   'libpq-dev python-dev'
]

### ENVIRONMENTS ###

def vagrant():
    """Defines the Vagrant virtual machine's environment variables.
    
    Environments define and contain the information need to SSH into a server,
    e.g. IP address, SSH key, username, and possibly password.
    """
    # Use Python's subprocess to run 'vagrant ssh-config' and parse results
    raw_ssh_config = subprocess.Popen(['vagrant', 'ssh-config'],
                                      stdout=subprocess.PIPE).communicate()[0]
    ssh_config = dict([l.strip().split() for l in raw_ssh_config.split("\n")
                       if l])
    env.hosts = ['127.0.0.1:%s' % (ssh_config['Port'])]
    env.user = ssh_config['User']
    env.key_filename = ssh_config['IdentityFile'].replace('"', '')
    env.virtualenv = {'dir': '/server', 'name': 'venv'}


def bootstrap():
    """Set up and configure Vagrant to be able to serve the web app.
    Runs commands on the command line to configure the Ubuntu server.
    """
    sub_install_packages()
    sub_install_virtualenv()
    sub_create_virtualenv()
    sub_install_python_requirements()


def sub_install_packages():
    """Install Ubuntu packages using apt-get, Ubuntu's package manager."""
    sudo('apt-get update')  # Update repository links
    sudo('apt-get -y upgrade')  # Upgrade the system
    package_str = ' '.join(INSTALL_PACKAGES)
    sudo('apt-get -y install ' + package_str)  # Install the packages


def sub_install_virtualenv():
    """Install the Python package 'virtualenv' so we can install Python
    packages safely into a virtualenv and not the system Python.
    """
    sudo('pip install virtualenv')  # Need sudo b/c installing to system Python


def sub_create_virtualenv():
    """Creates a Python virtualenv within which application requirements will
    be installed.
    """
    # Create folder to put virtualenv within
    mkdir = 'mkdir -p {0}; chown {1} {0}'.format(
       env.virtualenv['dir'], env.user)
    sudo(mkdir)
    # Create the virtualenv if it doesn't exist
    mkvenv = 'if [ ! -d {0}/{1} ]; then virtualenv {0}/{1}; fi'.format(    
       env.virtualenv['dir'], env.virtualenv['name'])
    run(mkvenv)


def sub_install_python_requirements():
    """Install the Flask apps' Python requirements into the virtualenv.
    We need to activate the virtualenv before installing into it. We do that
    with the command 'source /server/bin/activate'. The application requirements
    live in the requirements.txt file shared with the VM. This file lives at
    /vagrant/flask_ml/requirements.txt.
    """
    # Activate the virtualenv
    activate = 'source {0}/{1}/bin/activate'.format(
        env.virtualenv['dir'], env.virtualenv['name'])
    # Install Python requirements
    install = 'pip install -r /vagrant/hs-698-project/requirements.txt'
    # Join and execute the commands
    run(activate + '; ' + install)


def dev_server():
    """Run the Flask development server on the VM."""
    # Activate the virtualenv
    activate = 'source {0}/{1}/bin/activate'.format(
        env.virtualenv['dir'], env.virtualenv['name'])
    # Run the file run_api.py to start the Flask app
    dev_server = 'python /vagrant/hs-698-project/run_api.py'
    run(activate + '; ' + dev_server)
