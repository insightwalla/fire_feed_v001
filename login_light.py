from typing import Callable
import yaml
from yaml import SafeLoader
from streamlit_authenticator import Authenticate
import streamlit as st
from utils import *

config_file = "config.yaml"
def login(render_func):
    with open(config_file) as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    c1, c2 = st.columns(2)
    with c1:
        name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status: 
        with c1:
            authenticator.logout('Logout', 'main')

        c2.write(f'Welcome *{name}*')
        render_func()

    else: # no login attempt
        c1.warning('Please enter your username and password')