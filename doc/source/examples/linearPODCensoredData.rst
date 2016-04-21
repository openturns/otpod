
`ipynb source code <linearPODCensoredData.ipynb>`_

Linear model POD with censored data
===================================

.. code:: python

    # import relevant module
    import openturns as ot
    import otpod
    # enable display figure in notebook
    %matplotlib inline

Generate data
-------------

.. code:: python

    N = 100
    ot.RandomGenerator.SetSeed(123456)
    defectDist = ot.Uniform(0.1, 0.6)
    # normal epsilon distribution
    epsilon = ot.Normal(0, 1.9)
    defects = defectDist.getSample(N)
    signalsInvBoxCox = defects * 43. + epsilon.getSample(N) + 2.5
    # Inverse Box Cox transformation
    invBoxCox = ot.InverseBoxCoxTransform(0.3)
    signals = invBoxCox(signalsInvBoxCox)

Build POD using previous linear analysis
----------------------------------------

.. code:: python

    noiseThres = 60.
    saturationThres = 1700.
    
    # run the analysis with Gaussian hypothesis of the residuals (default case)
    analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres,
                                                   saturationThres, boxCox=True)

.. code:: python

    # signal detection threshold
    detection = 200.
    # Use the analysis to build the POD with Gaussian hypothesis
    # keyword arguments must be given
    PODGauss = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
    PODGauss.run()

Build POD with Gaussian hypothesis
----------------------------------

.. code:: python

    # The previous POD is equivalent to the following POD
    PODGauss = otpod.UnivariateLinearModelPOD(defects, signals, detection,
                                              noiseThres, saturationThres,
                                              resDistFact=ot.NormalFactory(),
                                              boxCox=True)
    PODGauss.run()

Compute detection size
----------------------

.. code:: python

    # Detection size at probability level 0.9
    # and confidence level 0.95
    print PODGauss.computeDetectionSize(0.9, 0.95)


.. parsed-literal::

    [a90 : 0.30373, a90/95 : 0.317848]


get POD NumericalMathFunction
-----------------------------

.. code:: python

    # get the POD model
    PODmodel = PODGauss.getPODModel()
    # get the POD model at the given confidence level
    PODmodelCl95 = PODGauss.getPODCLModel(0.95)
    
    # compute the probability of detection for a given defect value
    print 'POD : {:0.3f}'.format(PODmodel([0.3])[0])
    print 'POD at level 0.95 : {:0.3f}'.format(PODmodelCl95([0.3])[0])


.. parsed-literal::

    POD : 0.887
    POD at level 0.95 : 0.830


Show POD graph
--------------

Mean POD and POD at confidence level with the detection size for a given probability level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    fig, ax = PODGauss.drawPOD(probabilityLevel=0.9, confidenceLevel=0.95,
                          name='figure/PODGaussCensored.png')
    # The figure is saved in PODGauss.png
    fig.show()



.. image:: linearPODCensoredData_files/linearPODCensoredData_16_0.png


Build POD only with the filtered data
-------------------------------------

A static method is used to get the defects and signals only in the
uncensored area.

.. code:: python

    print otpod.DataHandling.filterCensoredData.__doc__


.. parsed-literal::

    
            Sort defects and signals with respect to the censore threholds.
    
            Parameters
            ----------
            defects : 2-d sequence of float
                Vector of the defect sizes.
            signals : 2-d sequence of float
                Vector of the signals, of dimension 1.
            noiseThres : float
                Value for low censored data. Default is None.
            saturationThres : float
                Value for high censored data. Default is None
    
            Returns
            -------
            defectsUnc : 2-d sequence of float
                Vector of the defect sizes in the uncensored area.
            defectsNoise : 2-d sequence of float
                Vector of the defect sizes in the noisy area.
            defectsSat : 2-d sequence of float
                Vector of the defect sizes in the saturation area.
            signalsUnc : 2-d sequence of float
                Vector of the signals in the uncensored area.
    
            Notes
            -----
            The data are sorted in three different vectors whether they belong to
            the noisy area, the uncensored area or the saturation area.
            


.. code:: python

    result = otpod.DataHandling.filterCensoredData(defects, signals,
                                                   noiseThres, saturationThres)
    defectsFiltered = result[0]
    signalsFiltered = result[3]

.. code:: python

    PODfilteredData = otpod.UnivariateLinearModelPOD(defectsFiltered, signalsFiltered,
                                                     detection,
                                                     resDistFact=ot.NormalFactory(),
                                                     boxCox=True)
    PODfilteredData.run()

.. code:: python

    # Detection size at probability level 0.9
    # and confidence level 0.95
    print PODfilteredData.computeDetectionSize(0.9, 0.95)


.. parsed-literal::

    [a90 : 0.295976, a90/95 : 0.310948]


.. code:: python

    fig, ax = PODfilteredData.drawPOD(probabilityLevel=0.9, confidenceLevel=0.95,
                          name='figure/PODGaussFiltered.png')
    # The figure is saved in PODGauss.png
    fig.show()



.. image:: linearPODCensoredData_files/linearPODCensoredData_22_0.png


