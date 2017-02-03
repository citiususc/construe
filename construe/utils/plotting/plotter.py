# -*- coding: utf-8 -*-
# pylint: disable-msg=C0103
"""
Module that contains utility functions to the graphical representation of
results.
"""

__author__ = "T. Teijeiro"
__date__ = "$30-nov-2011 18:01:49$"

import construe.knowledge.observables as o
import construe.knowledge.abstraction_patterns as ap
import construe.utils.pyperclip as pyperclip
from ..units_helper import msec2samples as m2s
from matplotlib.pyplot import figure
from matplotlib.patches import Ellipse
from matplotlib._pylab_helpers import Gcf
from collections import deque
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import pygraphviz as pgv
from math import sqrt
import numpy as np
import construe.acquisition.signal_buffer as sig_buf

LEVELS = ap.get_max_level() + 1

def _get_obs_descriptor(observation):
    """
    Obtains an observation descriptor (color, level) to represent each of
    the observations
    """
    colors = {}
    colors[o.PWave]           = ('#66A2A1', 0.2)
    colors[o.TWave]           = ('#E5FF00', 0.2)
    colors[o.RPeak]           = ('#0000FF', 1.0)
    colors[o.Normal_Cycle]    = ('#009C00', 0.2)
    colors[o.Sinus_Rhythm]    = ('#66A2A1', 0.2)
    colors[o.Cardiac_Rhythm]  = ('#66A2A1', 0.2)
    colors[o.Extrasystole]    = ('#897E00', 0.2)
    colors[o.Tachycardia]     = ('#6E0000', 0.2)
    colors[o.Bradycardia]     = ('#3F3F3F', 0.2)
    colors[o.RhythmBlock]     = ('#672E00', 0.2)
    colors[o.Asystole]        = ('#000000', 0.2)
    colors[o.Bigeminy]        = ('#FFAA00', 0.2)
    colors[o.Trigeminy]       = ('#00897E', 0.2)
    colors[o.Ventricular_Flutter] = ('#844089', 0.2)
    colors[o.Atrial_Fibrillation] = ('#404389', 0.2)
    colors[o.Couplet]         = ('#D70751', 0.2)
    colors[o.RhythmStart]     = ('#004008', 1.0)
    colors[o.Noise]           = ('#808080', 1.0)
    colors[o.RDeflection]     = ('#008000', 1.0)
    if isinstance(observation, o.QRS):
        col = '#0000FF' if not observation.paced else '#FF0000'
        colors[o.QRS] = (col, 0.2)
    if isinstance(observation, o.Deflection):
        lev = 0 if not observation.level else min(observation.level.values())
        colors[o.Deflection]       = ('#800000', max(0.8 - 0.2*lev, 0.1))
    try:
        clazz = type(observation)
        return colors[clazz] + (ap.get_obs_level(clazz), )
    except KeyError:
        #Por defecto, devolvemos gris semitransparente
        return ('#000000', 0.2, ap.get_obs_level(type(observation)))


def constraints_network_graphviz(interpretation, outfile):
    """
    Draws the constraints network of an interpretation using graphviz.

    Parameters
    ----------
    interpretation:
        Interpretation containing the related observations.
    outfile:
        File where the figure will be saved
    """
    lst = []
    pats = sorted(interpretation.patterns,
                     key = lambda p : -ap.get_obs_level(p.automata.Hypothesis))
    for pat in pats:
        for tnet in pat.temporal_constraints:
            lst.extend(tnet.get_constraints())
    G = pgv.AGraph(directed=True)
    G.graph_attr['fontsize'] = '7'
    G.node_attr['style'] = 'filled'
    G.node_attr['fixedsize'] = 'true'
    G.node_attr['width'] = '.85'
    G.node_attr['ordering'] = 'in'
    for const in lst:
        G.add_edge(id(const.va), id(const.vb))
    #Color assignment
    observations = interpretation.get_observations()
    for node in G.nodes_iter():
        for obs in observations:
            if node == str(id(obs.start)) or node == str(id(obs.end)):
                descriptor = _get_obs_descriptor(obs)
                node.attr['group'] = type(obs).__name__
                #We set some transparency to the color
                node.attr['color'] = descriptor[0] + 'C0'
                node.attr['label'] = (str(obs.earlystart)
                                                if node == str(id(obs.start))
                                                else str(obs.lateend))
                break
    G.layout(prog = 'dot', args='-Grankdir=LR')
    G.draw(outfile)


def plot_constraints_network(interpretation, with_labels = False, fig = None):
    """
    Draws the full temporal constraints network of a specific interpretation.
    """
    #List with all the temporal constraints of the network
    lst = []
    for pat in interpretation.patterns:
        for tnet in pat.temporal_constraints:
            lst.extend(tnet.get_constraints())
    G = nx.DiGraph()
    for const in lst:
        G.add_edge(const.va, const.vb)
    #Color assignment
    observations = interpretation.get_observations()
    colors = []
    labels = {}
    pos = {}
    for node in G.nodes_iter():
        for obs in observations:
            if node is obs.start or node is obs.end:
                colors.append(_get_obs_descriptor(obs)[0])
                x = obs.earlystart if node is obs.start else obs.lateend
                y = ap.get_obs_level(type(obs))
                pos[node] = (x, y)
                labels[node] = str(x) if with_labels else ''
                break
    fig = fig or figure()
    fig.clear()
    nx.draw(G, pos, fig.gca(), node_color=colors, labels=labels)


def parallel_plot(signals, fig = None):
    """
    Draws a list of signals, each one represented as a numpy array, in a
    graphical window, with each signal located over the next, in parallel,
    and sharing the X axis.

    Parameters
    ----------
    signals:
        Signal list

    Returns
    ---------
    out:
        Reference to the figure where the signals have been plotted.
    """
    #Creamos la figura, y la configuramos para que no haya espacio entre los
    #dibujos
    if fig is None:
        fig = figure()
    fig.clear()
    fig.subplots_adjust(hspace=0.001)
    for i in xrange(1, len(signals)+1):
        #Todas las gráficas excepto la primera compartirán los ejes
        shared_axes = None if i == 1 else fig.get_children()[1]
        fig.add_subplot(len(signals), 1, i,
                        sharex = shared_axes).plot(signals[i-1])
    return fig


def plot_observations(signal, interpretation, fig = None):
    """
    Draws a set of observations as filled areas over a background signal
    fragment. Each observable type is painted with a different colour, obtained
    through the _get_obs_descriptor() function.

    Parameters
    ----------
    signal:
        Signal fragment to represent.
    interpretation:
        Interpretation containing the observations to be plotted.

    Returns:
        Reference to the ObservationVisualizer used to draw.
    """
    obsview = ObservationVisualizer(signal, interpretation, fig)
    obsview.draw()
    return obsview


class ObservationVisualizer(object):
    """
    Class that encapsulates a figure, and the behaviour to plot specific
    observations and manage them.
    """
    def __init__(self, signal, interpretation, fig = None):
        self.signal = signal
        self.interpretation = interpretation
        self.fig = fig if fig is not None else figure()
        #Calculation of parameters
        self.sig_limits = (signal.min(), signal.max())
        self.lev_height = (self.sig_limits[1] - self.sig_limits[0]) / LEVELS
        #Definition of a polygon for each observation
        self.trapezs = {}
        observations = list(self.interpretation.get_observations()) + [
                                   o for o,_ in self.interpretation.focus._lst]
        for obs in observations:
            level = _get_obs_descriptor(obs)[2]
            bottom = self.sig_limits[0] + level * self.lev_height
            self.trapezs[obs] = [[obs.earlystart, bottom],
                                 [obs.lateend, bottom],
                                 [obs.earlyend, bottom + self.lev_height],
                                 [obs.latestart, bottom + self.lev_height]]
            for i in xrange(4):
                if self.trapezs[obs][i][0] == np.inf:
                    self.trapezs[obs][i][0] = len(self.signal)
        #Event listeners
        self.fig.canvas.mpl_connect('button_release_event', self.__onclick)

    def __onclick(self, event):
        """Manager to the click event on the figure."""
        #Search for click inside an observation
        if event.button == 3 and event.inaxes:
            trans = event.inaxes.transData
            for obs in self.interpretation.get_observations():
                trn = [trans.transform(point) for point in self.trapezs[obs]]
                trd = [(trn[0][0]-10, trn[0][1]), (trn[1][0]+10, trn[1][1]),
                       (trn[2][0]+10, trn[2][1]), (trn[3][0]-10, trn[3][1])]
                if _point_inside_polygon(event.x, event.y, trd):
                    plot = self.fig.get_children()[1]
                    #TODO get the score of the observation
                    plot.text(event.xdata, event.ydata, str(1.0))
                    self.fig.canvas.draw()
                    break
        elif event.button == 2:
            #We remove all score anotations
            del self.fig.gca().texts[LEVELS:]
            self.fig.canvas.draw()


    def draw(self):
        """Draws the figure"""
        self.fig.clear()
        self.fig.add_subplot(111).plot(self.signal, color= '#000000')
        plot = self.fig.get_children()[1]
        plot.set_axis_bgcolor('white')
        plot.yaxis.set_ticks([])
        #We obtain the figure limits to plot the observations
        min_sig = self.sig_limits[0]
        #Level annotation
        ypos = min_sig + self.lev_height / 2.0
        for i in xrange(LEVELS):
            plot.hlines(min_sig + i * self.lev_height, 0, len(self.signal),
                                                               color='#DCDCDC')
            plot.text(10, ypos, 'L' + str(i))
            ypos += self.lev_height
        #Observation drawing
        last_point = 0
        observations = list(self.interpretation.get_observations()) + [
                                   o for o,_ in self.interpretation.focus._lst]
        for obs in observations:
            color, alpha, level = _get_obs_descriptor(obs)
            if level > 0 and obs.lateend > last_point and obs.lateend < np.inf:
                last_point = obs.lateend
            #We fill the larger and shorter intervals of visualization
            plot.fill([pair[0] for pair in self.trapezs[obs]],
                      [pair[1] for pair in self.trapezs[obs]],
                      color= color, alpha= alpha)
            #And the time point of the variable
            bottom = self.sig_limits[0] + level * self.lev_height
            plot.plot([obs.time.start, obs.time.end],
                                 [bottom, bottom+self.lev_height], color=color)
        #Focus draw
        if self.interpretation.focus:
            focus = self.interpretation.focus.top[0]
            color, _, level = _get_obs_descriptor(focus)
            y = self.sig_limits[0] + (level+0.5) * self.lev_height
            xmin = min(focus.latestart, focus.earlyend)
            xmax = max(focus.latestart, focus.earlyend)
            ell = Ellipse(xy = [xmin + (xmax-xmin)/2, y],
                         width = max(m2s(250),
                                     min(len(self.signal)-focus.earlystart,
                                              focus.lateend-focus.earlystart)),
                                                      height = self.lev_height)
            ell.set_facecolor('none')
            ell.set_edgecolor(color)
            ell.set_linewidth(3.0)
            plot.add_artist(ell)
        #We set the X axes centered at the lateend
        self.fig.gca().set_xbound(lower= last_point-1000,
                                                       upper= last_point+1000)
        self.fig.gca().set_ybound(lower= min_sig-
                                               (self.sig_limits[1]-min_sig)/10,
                                  upper= self.sig_limits[1]+
                                               (self.sig_limits[1]-min_sig)/10)
        self.fig.show()


def _point_inside_polygon(x, y, poly):
    """
    determine if a point is inside a given polygon or not
    Polygon is a list of (x,y) pairs.
    """
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in xrange(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def plot_branch(interpretation, fig=None, label_funcs=None, target=None,
                full_tree=False):
    """
    Draws a tree of interpretations, starting for the one passed as
    argument to all the subinterpretations in the hierarchy.

    Parameters
    ----------
    interpretation:
        Root interpretation to visualize
    fig:
        Figure where the graph is shown
    label_funcs:
        Dictionary to map characters with functions to draw node labels. When
        a keyboard event is detected and the character is in the keyset, the
        graph is replotted using the corresponding function. By default, the
        first function is used, or no labels are plot.
    target:
        Node whose trace is remarked in the plot. The path from this node to
        the root node is plotted in a different color.
    full_tree:
        If true, the full tree is drawn on creation, and all nodes are plot. If
        false, the nodes are plotted on demand.
    """
    view = InterpretationVisualizer(interpretation, fig, label_funcs, target,
                                    full_tree)
    view.draw()
    return view

class InterpretationVisualizer(object):
    """
    Class that encapsulates a figure, and the behaviour to dynamically plot
    interpretations.
    """

    def __init__(self, interp, fig=None, label_funcs=None, target=None,
                 full_tree=False):
        """
        Creates a new instance, associating it to a figure, where the
        branches will be visualized. See *plot_branch* for details on
        arguments.
        """
        self._fig = fig if fig != None else figure()
        self._subfigs = {}
        self.pos = None
        self.drnodes = set([interp])
        self.toremark = set()
        if target is not None:
            while True:
                self.toremark.add(target)
                if target is interp:
                    break
                target = target.parent
        #Graph creation and setting
        self.graph = nx.DiGraph()
        self.graph.add_node(interp)
        _add_subbranches(interp, self.graph)
        #HINT These lines add all nodes to the graph
        if full_tree:
            stack = [interp]
            while stack:
                n = stack.pop()
                self.drnodes.add(n)
                stack.extend(self.graph[n].keys())
        self.labels = {}
        labelling = label_funcs or dict({'e': lambda node: ''})
        for key, func in labelling.iteritems():
            self.labels[key] = {}
            for node in self.graph.nodes_iter():
                self.labels[key][node] = func(node)
        #Horizontal tree layout
        self.pos = graphviz_layout(self.graph, prog='dot', args='-Grankdir=LR')
        #Event listener
        self._fig.canvas.mpl_connect('button_release_event', self.__onclick)
        self._fig.canvas.mpl_connect('key_release_event', self.__onkey)


    def __onclick(self, event):
        """Manager to the click event on the figure."""
        #Left click
        if event.button in (1, 2, 3) and event.inaxes:
            trans = event.inaxes.transData
            #Distance from nodes (in pixels)
            for node in self.drnodes.copy():
                posit = self.pos[node]
                xn, yn = trans.transform(posit)
                dx, dy = (event.x-xn, event.y-yn)
                dist = sqrt(dx*dx + dy*dy)
                #Nodes have a 10-pixel radius
                if dist < 10:
                    #The clicked button will define the action
                    #Button 1: plot
                    #If the branch is already plot, we just show it
                    if event.button == 1:
                        if node in self._subfigs:
                            mgr = Gcf.get_fig_manager(
                                                self._subfigs[node].fig.number)
                            mgr.window.activateWindow()
                            mgr.window.raise_()
                        else:
                            #Else plot the observations
                            signal = sig_buf.get_signal(sig_buf.
                                                      get_available_leads()[0])
                            #We have to keep a reference to the object to avoid
                            #garbage collection and loosing the event manager
                            #see http://matplotlib.org/users/event_handling.html
                            obsview = ObservationVisualizer(signal, node)
                            mgr = Gcf.get_fig_manager(obsview.fig.number)
                            mgr.window.move(0, 0)
                            obsview.fig.canvas.set_window_title(str(node))
                            self._subfigs[node] = obsview
                            obsview.draw()
                    #Button 2: Add child nodes of the selected one to the plot.
                    elif event.button == 2:
                        pyperclip.copy(str(node))
                        stack = [node]
                        while stack:
                            n = stack.pop()
                            self.drnodes.add(n)
                            if n is node or not n.is_firm:
                                stack.extend(self.graph[n].keys())
                        self.redraw()
                    #Button 3: Copy the branch name to the clipboard
                    elif event.button == 3:
                        pyperclip.copy(str(node))

    def __onkey(self, event):
        """Manager for the key press event on the figure."""
        #We check if the key is in the functions dictionary
        if event.key in self.labels:
            labels_dict = self.labels[event.key]
            self.redraw(ldict=labels_dict)

    def redraw(self, ldict= None):
        """
        Updates the plot.
        """
        ldict = ldict or self.labels.values()[0]
        axes = self._fig.gca()
        #Save the axis limits
        x, y = axes.get_xlim(), axes.get_ylim()
        #Clear and repaint
        axes.clear()
        #Node selection and drawing
        colors = []
        for node in self.drnodes:
            if node.is_firm:
                colors.append('#348ABD')
            elif node in self.toremark:
                colors.append('#8EBA42')
            else:
                colors.append('#E24A33')
        nx.draw(self.graph, self.pos, axes, nodelist = self.drnodes,
                                       edgelist=self.graph.edges(self.drnodes),
                                           node_color = colors, labels = ldict)
        #Limit reset
        axes.set_xlim(x)
        axes.set_ylim(y)
        self._fig.show()


    def draw(self):
        """
        Draws a tree of PatternBranch objects, starting for the one passed as
        argument to all the subbranches in the hierarchy.
        """
        self._fig.clear()
        self._fig.add_axes([0, 0, 1, 1])
        colors = []
        for node in self.drnodes:
            if node.is_firm:
                colors.append('#348ABD')
            elif node in self.toremark:
                colors.append('#8EBA42')
            else:
                colors.append('#E24A33')
        nx.draw(self.graph, self.pos, self._fig.gca(), nodelist = self.drnodes,
                                       edgelist=self.graph.edges(self.drnodes),
                                                           node_color = colors)
        self.redraw()


def _add_subbranches(interp, graph):
    """Recursively adds subbranches of an interpretation to a graph"""
    queue = deque([interp])
    while queue:
        head = queue.popleft()
        for chl in head.child:
            graph.add_node(chl)
            graph.add_edge(head, chl)
            queue.append(chl)

if __name__ == "__main__":
    pass
