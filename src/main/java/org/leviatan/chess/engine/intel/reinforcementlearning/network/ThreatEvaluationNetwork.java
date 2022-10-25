package org.leviatan.chess.engine.intel.reinforcementlearning.network;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.leviatan.chess.board.Bando;
import org.leviatan.chess.board.Tablero;
import org.leviatan.chess.engine.deeplearning.DeepLearningUtils;
import org.leviatan.chess.engine.deeplearning.DenseNetworkInputGenerator;
import org.leviatan.chess.engine.deeplearning.NetworkInputGenerator;
import org.leviatan.chess.tools.platform.AppLogger;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * ThreatEvaluationNetwork.
 *
 * Red que evalua la amenaza que supone el bando Blanco sobre el bando Negro.
 *
 * @author Alejandro
 *
 */
public class ThreatEvaluationNetwork {

    final NetworkInputGenerator networkInputGenerator = new DenseNetworkInputGenerator();

    /** NUM_OUTPUTS. */
    public static final List<Integer> ARRAY_SIZE_INNER_LAYERS = Arrays.asList(10000, 5000, 1000, 300);

    /** NUM_OUTPUTS. */
    public static final int NUM_OUTPUTS = 1;

    private final String pathDirStoreLoad;

    private final MultiLayerNetwork net;

    private int getNumInputs() {
        return this.networkInputGenerator.getInputSize();
    }

    private int getNumOutputs() {
        return NUM_OUTPUTS;
    }

    private String getPathModel() {
        return this.pathDirStoreLoad + "/" + getModelFileName();
    }

    private String getModelFileName() {
        return this.getClass().getSimpleName() + ".model";
    }

    /**
     * Constructor for ThreatEvaluationNetwork.
     *
     * @param pathDirStoreLoad
     *            pathStoreLoad
     * @throws IOException
     */
    public ThreatEvaluationNetwork(final String pathDirStoreLoad) throws IOException {

        this.pathDirStoreLoad = pathDirStoreLoad;
        final String pathModel = getPathModel();

        final File fileModel = new File(pathModel);

        if (fileModel.exists()) {
            this.net = ModelSerializer.restoreMultiLayerNetwork(pathModel, true);
            AppLogger.logDebug("Modelo cargado: " + pathModel);

        } else {

            final MultiLayerConfiguration conf = buildNetwork();

            this.net = new MultiLayerNetwork(conf);
            this.net.init();
            AppLogger.logDebug("Modelo nuevo");
        }

        this.net.setListeners(new ScoreIterationListener(100));
    }

    private MultiLayerConfiguration buildNetwork() {

        final int numInputs = getNumInputs();
        final int numOutputs = getNumOutputs();

        final int seed = 12345;
        final double learningRate = 0.00001;

        ListBuilder listBuilder = new NeuralNetConfiguration.Builder().seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Nesterovs(learningRate, 0.9)).list();

        int numInputNextLayer = numInputs;
        int numOutputNextLayer = -1;

        for (int i = 0; i < ARRAY_SIZE_INNER_LAYERS.size(); i++) {

            numOutputNextLayer = ARRAY_SIZE_INNER_LAYERS.get(i);

            listBuilder = addHiddenLayer(i, listBuilder, numInputNextLayer, numOutputNextLayer);

            numInputNextLayer = numOutputNextLayer;
        }

        final MultiLayerConfiguration conf = listBuilder.layer(ARRAY_SIZE_INNER_LAYERS.size(),
                new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER).nIn(numInputNextLayer).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();

        return conf;
    }

    private ListBuilder addHiddenLayer(final int index, final ListBuilder listBuilder, final int numImput, final int numOutput) {

        final Layer layer = new DenseLayer.Builder().nIn(numImput).nOut(numOutput).weightInit(WeightInit.XAVIER).activation(Activation.RELU)
                .build();
        return listBuilder.layer(index, layer);
    }

    /**
     * Entrena la red.
     *
     * @param numEpochs
     *            numEpochs
     * @param listTableroAndThreat
     *            listTableroAndThreat
     * @throws IOException
     */
    public void train(final int numEpochs, final List<TableroAndThreat> listTableroAndThreat) throws IOException {

        final DataSetIterator dataSetIteratorBlancas = new ThreatDenseDatasetIterator(listTableroAndThreat, Bando.BLANCO);
        final DataSetIterator dataSetIteratorNegras = new ThreatDenseDatasetIterator(listTableroAndThreat, Bando.NEGRO);

        for (int i = 0; i < numEpochs; i++) {

            this.net.fit(dataSetIteratorBlancas);
            this.net.fit(dataSetIteratorNegras);

            dataSetIteratorBlancas.reset();
            dataSetIteratorNegras.reset();
        }

        final String pathModel = getPathModel();
        ModelSerializer.writeModel(this.net, pathModel, true);
    }

    public double getThreatOfGivenBando(final Tablero tablero, final Bando bando) {

        if (Bando.BLANCO.equals(bando)) {
            return getThreat(tablero);
        } else {
            return getThreat(tablero.creaTableroEspecular());
        }
    }

    private double getThreat(final Tablero tablero) {

        final double[] inputByteArray = this.networkInputGenerator.getInput(tablero);
        final INDArray input = DeepLearningUtils.getINDArray(inputByteArray);

        final INDArray networkOutput = this.net.output(input);

        return networkOutput.getDouble(0);
    }

}
