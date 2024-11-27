# -*- coding: utf-8 -*-
"""Module for calculating values to be plotted in a webviz
   TornadoPlot for one response and one design set up
"""
import pandas as pd


def real_mask(dfr, start, end):
    """ Creates mask for which realisations to calc from """
    mask = (dfr.REAL >= start) & (dfr.REAL <= end)
    return mask


def check_selector(resultfile, selector):
    """ Checks whether selector is in resultfile headers """
    if selector not in resultfile.columns:
        raise ValueError("Did not find ", selector, " as column in",
                         "resultfile")


def check_selection(resultfile, selector, selection):
    """ Checks whether selction is in resultfile values
        in the column selector """
    for sel in selection:
        if sel not in resultfile[selector].values:
            if str(sel).strip('[]').lower() == 'total':
                raise ValueError("Use 'all' to sum all", selector,
                                 "not 'total'")
            else:
                raise ValueError("Selection ", sel,
                                 " not found in column",
                                 selector, "in resultfile")


def check_response(resultfile, response):
    """ Checks whether respones in in resultfile """
    if response not in resultfile.columns:
        raise ValueError("Did not find ", response, " as column in",
                         "resultfile")


def cut_by_seed(tornadotable):
    """ Removes sensitivities smaller than seed sensitivity from table """
    maskseed = tornadotable.sensname == 'seed'
    seedlow = tornadotable[maskseed].low.abs()
    seedhigh = tornadotable[maskseed].high.abs()
    seedmax = max(float(seedlow), float(seedhigh))
    dfr_filtered = (tornadotable.loc[
        (tornadotable['sensname'] == 'seed') |
        ((tornadotable['low'].abs() >= seedmax) |
         (tornadotable['high'].abs() >= seedmax))])
    return dfr_filtered


def calc_tornadoinput(designsummary, resultfile, response, selectors,
                      selection, reference='seed', scale='percentage',
                      cutbyseed=False):
    """
     Calculates values to be plotted in a webviz TornadoPlot for one response
     and one design set up

    :returns tornadoinput: dataframe on format for webviz.TornadoPlot class
    :returns  ref_average: average for reference sensitivity
    :param designsummary: pandas dataframe with summary of designmatrix
    :param resultfile: a pandas dataframe with all results collected
     and 'REAL' as one of the headers
    :param response(str): name of response in resultfile to plot tornado for
    :param selectors(list of strings): which selectors to choose/filter on
    :param selections(list of lists): what to filter on for each selector
    :param reference: specifying what is the reference for the tornado plots
     valid choices are 'seed' or a realisation number
    :param scale: whether to plot absolute numbers or percentage compared
     to reference mean valid choices are 'percentage' and 'absolute'
    :param cutbyseed: If True sensitivities smaller than seed are excluded
    """
    # Check that chosen response exists in resultfile
    check_response(resultfile, response)

    # Filter on chosen selectors(column names) and selections(values)
    for itr, sel in enumerate(selection):
        header = selectors[itr]
        sel_stripped = str(sel).strip('[]').lower()
        check_selector(resultfile, selectors[itr])
        if sel_stripped != "'all'":
            check_selection(resultfile, selectors[itr], sel)
            resultfile = resultfile[(resultfile[header].isin(sel))]

    # summing over chosen selections for each realisation
    headers = ['REAL'] + selectors
    resultcopy = resultfile.copy()
    resultcopy.set_index(headers, inplace=True)
    resultcopy.sort_index(inplace=True)
    dfr_summed = resultcopy.groupby(['REAL']).sum()
    dfr_summed.reset_index(inplace=True)

    # Creates empty dfr to store stats in, webportal tornadoplot input format
    tornadoinput = pd.DataFrame(columns=['sensno', 'sensname', 'low',
                                         'high', 'leftlabel', 'rightlabel',
                                         'numreal1', 'numreal2'])

    # Calculate mean for reference
    if reference == 'seed':
        startreal = int(designsummary[
            designsummary['sensname'] == 'seed']['startreal1'])
        endreal = int(designsummary[
            designsummary['sensname'] == 'seed']['endreal1'])
        mask = real_mask(dfr_summed, startreal, endreal)
        ref_avg = dfr_summed[mask][response].mean()
    elif reference.isdigit():
        if int(reference) >= 0 and int(reference) <= dfr_summed.REAL.max():
            startreal = int(reference)
            endreal = int(reference)
            mask = real_mask(dfr_summed, startreal, endreal)
            ref_avg = dfr_summed[mask][response].mean()
    else:
        raise ValueError("Reference should be 'seed' or a real number",
                         reference)

    # for each sensitivity calculate statistics
    for sensno in range(len(designsummary)):
        sensname = designsummary.loc[sensno]['sensname']
        startreal = designsummary.loc[sensno]['startreal1']
        endreal = designsummary.loc[sensno]['endreal1']
        mask = real_mask(dfr_summed, startreal, endreal)
        numreal1 = len(dfr_summed.REAL[mask])
        avg1 = dfr_summed[mask][response].mean()-ref_avg
        if designsummary.loc[sensno]['senstype'] == 'mc':
            p90 = dfr_summed[mask][response].quantile(0.10)-ref_avg
            p10 = dfr_summed[mask][response].quantile(0.90)-ref_avg
            subset1name = 'p90'
            subset2name = 'p10'
            tornadoinput.loc[sensno] = [sensno, sensname,
                                        p90, p10,
                                        subset1name, subset2name,
                                        numreal1, numreal1]
        elif designsummary.loc[sensno]['senstype'] == 'scalar':
            subset1name = designsummary.loc[sensno]['casename1']
            # if case 2 exists
            if designsummary.loc[sensno]['casename2'] is not None:
                startreal = designsummary.loc[sensno]['startreal2']
                endreal = designsummary.loc[sensno]['endreal2']
                mask = real_mask(dfr_summed, startreal, endreal)
                numreal2 = len(dfr_summed.REAL[mask])
                avg2 = dfr_summed[mask][response].mean()-ref_avg
                subset2name = designsummary.loc[sensno]['casename2']
            else:
                avg2 = 0
                numreal2 = 0
                subset2name = None
            # If case1 has a higher value than case 2, swap values and names
            if avg2 < avg1:
                avg1, avg2 = avg2, avg1
                numreal1, numreal2 = numreal2, numreal1
                subset1name, subset2name = subset2name, subset1name
            tornadoinput.loc[sensno] = [sensno, sensname, avg1, avg2,
                                        subset1name, subset2name,
                                        numreal1, numreal2]
        else:
            raise ValueError("Sensitivity type should be 'mc' or 'scalar': "
                             + designsummary.loc[sensno]['senstype'] + '\n'
                             + "Something wrong with designsummary?")

    tornadoinput['true_low'] = (tornadoinput['low']+ref_avg).astype(int)
    tornadoinput['true_high'] = (tornadoinput['high']+ref_avg).astype(int)

    if scale == 'percentage':
        if ref_avg != 0:
            tornadoinput['low'] = 100*tornadoinput['low']/ref_avg
            tornadoinput['high'] = 100*tornadoinput['high']/ref_avg
        else:
            tornadoinput['low'] = 0
            tornadoinput['high'] = 0

    # Drops sensitivities smaller than seed if specified
    if cutbyseed and tornadoinput['sensname'].str.contains('seed').any():
        tornadoinput = cut_by_seed(tornadoinput)

    # Return results that can be used for plotted in e.g. webviz
    tornadoinput = tornadoinput.drop(['sensno'], axis=1).set_index('sensname')

    return (tornadoinput, ref_avg)
