from setuptools import setup, Extension

setup(name="tsumegoboard", version="1.0", ext_modules=[Extension("tsumegoboard", ["tsumegoboard.cpp"])])
