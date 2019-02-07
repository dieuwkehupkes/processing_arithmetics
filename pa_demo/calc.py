from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

import io
import random
import base64
from flask import Response
import matplotlib
matplotlib.use("Agg")

from processing_arithmetics.arithmetics.treebanks import treebank
from processing_arithmetics.arithmetics.MathExpression import MathExpression
from processing_arithmetics.arithmetics.MathTreebank import MathTreebank
from processing_arithmetics.sequential.architectures import ScalarPrediction, DiagnosticClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
import re
import pandas as pd
from plotnine import *
import plotnine
import matplotlib
from collections import OrderedDict

bp = Blueprint('calc', __name__)

digits = np.arange(-10,11)
operators = ['+', '-']

predictor_map = OrderedDict([
        ('intermediate_recursively', "subtotal_recursive"),
        ('intermediate_locally', "subtotal_cumulative"),
    ])
predictors = predictor_map.keys()

classifier_map = OrderedDict([
        ('grammatical', "grammatical"),
        ('subtracting', "mode"),
        ('switch_mode', "switch_mode"),
        ('depth', "depth"),
        ('minus1depth', "minus1depth"),
        ('minus2depth', "minus2depth"),
        ('minus3depth', "minus3depth"),
        ('minus4depth', "minus4depth"),
        ('minus1depth_count', "minus1depth_count"),
    ])

full_map = classifier_map.copy()
full_map.update(predictor_map)

classifier_comments = {
    'intermediate_recursively': "This real-valued feature represents the intermediate outcome at every point in time, assuming that this outcome is computed using the <em>recursive strategy</em>.",
    'intermediate_locally': "This real-valued feature represents the intermediate outcome of the <em>cumulative strategy</em>.",
    'grammatical': "This binary feature represents whether an expression is grammatical (this will thus only be the case at the end of the expression, when all brackets are closed)",
    'subtracting': "This binary feature is relevant for the <em>cumulative strategy</em>, it expresses whether the next feature should be added (1) or subtracted (0).",
    'mode_switch': "This binary feature describes whether the <cite>mode</cite> feature remains the same, or changes.",
    'minus1depth': "This binary feature represents whether the representation is within the scope of <em>at least 1 minus</em> (in other words, this feature is true when a leaf node has at least one ancestor node which is a minus).",
    'minus2depth': "Similar to <cite>minus1depth</cite>, but for <strong>2</strong> minusses",
    'minus3depth': "Similar to <cite>minus1depth</cite>, but for <strong>3</strong> minusses",
    'minus4depth': "Similar to <cite>minus1depth</cite>, but for <strong>4</strong> minusses",
    'minus1depth_count': "Keeping track of the minusdepth of a sentence requires <em>counting</em>, this real-valued feature stores how many brackets should still be closed for the minus1depth to go to 0.",
    }
all_clas = "subtracting intermediate_locally intermediate_recursively grammatical depth minus1depth minus2depth minus3depth minus4depth minus1depth_count switch_mode".split()


def model_path(modelname):
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    model_path = os.path.join(SITE_ROOT, 'models', modelname)
    return model_path

arch1 = ScalarPrediction(digits=digits, operators=operators)
arch1.add_pretrained_model(model=model_path("ScalarPrediction_GRU_infix_2018-10-11T11+02:00_10.h5"))
arch1.model.compile(loss=arch1.loss_functions, optimizer='adam', metrics=arch1.metrics, loss_weights=arch1.loss_weights)

dcarch1 = DiagnosticClassifier(
        digits=digits,
        operators=operators,
        classifiers=all_clas)
dcarch1.add_pretrained_model(model=model_path("ScalarPrediction_GRU_infix_2018-10-11T11+02:00_10_dc8.h5"))
dcarch1.model.compile(
        loss=dcarch1.loss_functions,
        optimizer='adam',
        metrics=dcarch1.metrics,
        loss_weights=dcarch1.loss_weights)

#arch2 = ScalarPrediction(digits=digits, operators=operators)
#arch2.add_pretrained_model(model=model_path("./ScalarPrediction_GRU_infix_2018-10-11T11+02:00_16.h5"))
#arch2.model.compile(loss=arch2.loss_functions, optimizer='adam', metrics=arch2.metrics, loss_weights=arch2.loss_weights)
#
#dcarch2 = DiagnosticClassifier(
#        digits=digits,
#        operators=operators,
#        classifiers=all_clas)
#dcarch2.add_pretrained_model(model=model_path("ScalarPrediction_GRU_infix_2018-09-27T10+02:00_16_dc8.h5"))
#dcarch2.model.compile(
#        loss=dcarch2.loss_functions,
#        optimizer='adam',
#        metrics=dcarch2.metrics,
#        loss_weights=dcarch2.loss_weights)

def calculate_predictions(tb):
    def calculate_error_metrics( g ):
	rmse = np.sqrt( mean_squared_error( g['expected'], g['prediction'] ) )
	mae = mean_absolute_error(g['expected'], g['prediction'])
	return pd.Series( dict(  mae = mae, rmse = rmse ) )

    data = arch1.generate_test_data(tb, digits=digits)
    predictions = {}
    for name, X, Y in data:
        predictions[name] = {'actual value': Y['output'][0]}
        for n, arch in [("model #10", arch1)]: #, ("model #16", arch2)]:
            result = np.array(arch.model.predict(X))
            result = result.reshape(*result.shape[:-1])
            predictions[name][n] = result

    predictions = (pd.DataFrame(predictions['test treebank']).
                        melt(value_vars=['actual value', 'model #10'], #, 'model #16'],
                             value_name="prediction",
                             var_name="model"
                            )
                )
    dcdata = dcarch1.generate_test_data(tb, digits=digits)
    results = {}
    for name, X, Y in dcdata:
        results[name] = {}
        for n, arch in [("model", dcarch1)]: #, ("model #16", dcarch2)]:
            result = np.array(arch.model.predict(X))
            result = result.reshape(*result.shape[:-1])
            results[name][n] = result

    results = results['test treebank']

    plot_data = []
    for model in results:
        model_results = results[model]
        for n, example in enumerate(tb.examples):
            seq_len = len(list(example[0].iterate('infix')))
            for i, classifier in enumerate(all_clas):
                plot_data.append(pd.DataFrame({
                    'model': model,
                    'example': n,
                    'classifier': classifier,
                    'class_name': full_map[classifier],
                    'prediction': model_results[i, n, -seq_len:],
                    'expected': dcdata[0][2][classifier][n,-seq_len:,0],
                }).reset_index())
    plot_df = pd.concat(plot_data).reset_index(drop=True)
    plot_df['classifier'] = pd.Categorical(plot_df['classifier'])
    plot_df['class_name'] = pd.Categorical(plot_df['class_name'])
    errors = (plot_df.groupby(['classifier']) #, 'model'])
                    .apply(calculate_error_metrics)
                    .round(2)
             )
    plot_df = plot_df.melt(id_vars=['index', 'class_name', 'classifier', 'example', 'model'], value_vars=['expected', 'prediction'])

    return plot_df, predictions, errors

def plot_prediction_data(plot_df, tb, classifier):
    plot_df = plot_df[plot_df['classifier'] == classifier]

    plotnine.options.figure_size = (11,2)
    the_plot = ggplot(plot_df, aes(x="index", y="value")) + \
        geom_step(aes(color="variable", linetype="variable"), direction="vh", size=1, alpha=0.5) + \
        geom_point(aes(shape="variable"), size=1, alpha=0.5) + \
        scale_x_continuous(breaks=range(len(list(tb.examples[0][0].iterate('infix')))), labels=list(tb.examples[0][0].iterate('infix'))) + \
        scale_color_grey(start=0.6, end=0.4) + \
        theme_light() + \
        labs(y="") #+ \
        #facet_wrap("~model")

    if classifier in ["intermediate_locally", "intermediate_recursively"]:
        loc_df = (plot_df[(plot_df['index'] == plot_df['index'].max()) &
                         (plot_df['variable'] == "prediction")][['model', 'index', 'value']].
                    reset_index(drop=True))
        the_plot = the_plot + \
                geom_label(
                        mapping=aes(x="index", y="value", label="value"),
                        data=loc_df,
                        ha="right",
                        nudge_x=-0.5,
                        size=6,
                        alpha=0.7,
                        format_string="final: {:.02f}"
                    )
    if classifier not in ["intermediate_locally", "intermediate_recursively", "depth", "minus1depth_count"]:
        the_plot = the_plot + scale_y_continuous(
                limits = [0,1.1],
                breaks=np.linspace(0,1,11),
            )

    fig = the_plot.draw()
    return fig


@bp.route('/', methods=('POST','GET'))
def create():
    expression = None
    if request.method == 'POST':
        expression = request.form['expression']
        error = None

        if expression is None:
            error = 'Expression is required.'

        tb = MathTreebank({}, digits=digits)
        tokens = re.findall(r"(-?\d+|\(|\)|\+|\-)", expression)
        try:
            tb.add_example_from_string(" ".join(tokens))
        except ValueError:
            error = "Sorry, that doesn't seem to be a valid expression"

        if error is not None:
            flash(error)

    plots = {}
    predictions = None
    errors = None
    if request.method == 'POST' and error is None:
        df, predictions, errors = calculate_predictions(tb)
        predictions = predictions.round({'prediction': 2}).set_index("model").to_html(border=0)
        for classifier in full_map.keys():
            fig = plot_prediction_data(df, tb, classifier)
            output = io.BytesIO()
            fig.savefig(output, format="png")
            output.seek(0)
            graph_url = base64.b64encode(output.getvalue()).decode()
            plots[classifier] = 'data:image/png;base64,{}'.format(graph_url)
            matplotlib.pyplot.close(fig)

    if expression is None:
        expression = ""

    return render_template('calc/run.html', expression=expression, clas_map=classifier_map, pred_map=predictor_map, plots=plots, comments=classifier_comments, predictions=predictions, errors=errors)

