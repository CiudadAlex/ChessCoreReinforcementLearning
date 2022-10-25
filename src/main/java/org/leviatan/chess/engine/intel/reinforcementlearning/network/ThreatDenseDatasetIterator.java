package org.leviatan.chess.engine.intel.reinforcementlearning.network;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;

import org.leviatan.chess.board.Bando;
import org.leviatan.chess.engine.deeplearning.DeepLearningUtils;
import org.leviatan.chess.engine.deeplearning.DenseNetworkInputGenerator;
import org.leviatan.chess.engine.deeplearning.NetworkInputGenerator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class ThreatDenseDatasetIterator implements DataSetIterator {

    private static final long serialVersionUID = 1655984149540148845L;

    private final List<TableroAndThreat> listTableroAndThreat;
    private final Bando bando;

    private Iterator<TableroAndThreat> iterTableroAndThreat;

    private final NetworkInputGenerator networkInputGenerator = new DenseNetworkInputGenerator();
    private final int numInputs = this.networkInputGenerator.getInputSize();
    private final int numOutputs = 1;

    /**
     * Constuctor for ThreatDenseDatasetIterator.
     *
     * @param listTableroAndThreat
     *            listTableroAndThreat
     */
    public ThreatDenseDatasetIterator(final List<TableroAndThreat> listTableroAndThreat, final Bando bando) {

        this.listTableroAndThreat = listTableroAndThreat;
        this.bando = bando;

        this.iterTableroAndThreat = this.listTableroAndThreat.iterator();
    }

    @Override
    public boolean hasNext() {
        return this.iterTableroAndThreat.hasNext();
    }

    @Override
    public DataSet next() {
        return next(1);
    }

    @Override
    public DataSet next(final int num) {

        final List<TableroAndThreat> listTableroAndThreatBatch = new ArrayList<TableroAndThreat>();

        for (int i = 0; i < num; i++) {

            if (this.iterTableroAndThreat.hasNext()) {
                listTableroAndThreatBatch.add(this.iterTableroAndThreat.next());
            }
        }

        final Function<TableroAndThreat, double[]> extractorInput = tt -> this.networkInputGenerator
                .getInput(Bando.BLANCO == this.bando ? tt.getTablero() : tt.getTablero().creaTableroEspecular());
        final Function<TableroAndThreat, double[]> extractorOutput = tt -> new double[] {
                Bando.BLANCO == this.bando ? tt.getThreatBlancas() : tt.getThreatNegras() };

        return DeepLearningUtils.buildDataSet(listTableroAndThreatBatch, extractorInput, extractorOutput, this.numInputs, this.numOutputs);
    }

    @Override
    public int inputColumns() {
        return this.numInputs;
    }

    @Override
    public int totalOutcomes() {
        return this.numOutputs;
    }

    @Override
    public void reset() {
        this.iterTableroAndThreat = this.listTableroAndThreat.iterator();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public int batch() {
        return 1;
    }

    @Override
    public void setPreProcessor(final DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not implemented");
    }

}
