#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from visdom import Visdom
from typing import List, Union


class VisShow(object):
    def __init__(self, server, port, envdir, subenv):
        self.vis = Visdom(server, port=port, env=f'{envdir}_{subenv}')

    def update(self, target: str, X: List[Union[int, float]], Y: List[Union[int, float]]) -> None:
        attrname = f'__{target}'
        if hasattr(self, attrname):
            self.vis.line(Y, X, win=getattr(self, attrname), update='append')
        else:
            setattr(self, attrname, self.vis.line(
                Y, X, opts={'title': target}))
