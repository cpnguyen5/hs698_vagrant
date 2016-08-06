from fabric.api import run, sudo, env, cd
import subprocess
import fabtools

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
   'libpq-dev python-dev',
   'postgresql'
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


def aws():
    """Defines the AWS server's environment variables."""
    env.hosts = 'ec2-54-186-163-254.us-west-2.compute.amazonaws.com'
    env.user = 'ubuntu'
    env.key_filename = '/home/cpnguyen/usf/hs698_key/hs698project2.pem'
    env.virtualenv = {'dir': '/server', 'name': 'venv'}


def bootstrap():
    """Set up and configure Vagrant to be able to serve the web app.
    Runs commands on the command line to configure the Ubuntu server.
    """
    sub_install_packages()
    sub_install_virtualenv()
    sub_create_virtualenv()
    # sub_install_python_requirements()


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
    # install = 'pip install -r /vagrant/hs-698-project/requirements.txt'
    install = 'pip install -r /home/ubuntu/hs698_vagrant/hs-698-project/requirements.txt'
    # Join and execute the commands
    run(activate + '; ' + install)


def dev_server():
    """Run the Flask development server on the VM."""
    # Activate the virtualenv
    activate = 'source {0}/{1}/bin/activate'.format(
        env.virtualenv['dir'], env.virtualenv['name'])
    # Run the file run_api.py to start the Flask app
    # dev_server = 'python /vagrant/hs-698-project/run_api.py'
    dev_server = 'python /home/ubuntu/hs698_vagrant/hs-698-project/run_api.py'
    run(activate + '; ' + dev_server)


def postgres():
    """Initiate postgresql database instance."""
    # try:
    #     conn = psycopg2.connect("dbname='cms_post' user='postgres' host='localhost' password='abcd1234'")
    # except:
    #     print "Fail"

    # Activate the virtualenv
    # activate = 'source {0}/{1}/bin/activate'.format(
    #     env.virtualenv['dir'], env.virtualenv['name'])
    # Create db user if not yet exist
    print fabtools.postgres.user_exists('postgres1')
    if not fabtools.postgres.user_exists('postgres1'):
        fabtools.postgres.create_user('postgres1', password='abcd1234')

    # Create db if not yet exist
    print fabtools.postgres.database_exists('cms_post1')
    if not fabtools.postgres.database_exists('cms_post1'):
        fabtools.postgres.create_database('cms_post1', owner='postgres1')
    create_path = '/home/ubuntu/hs698_vagrant/hs-698-project/db_create.py'
    run('python ' + create_path)

