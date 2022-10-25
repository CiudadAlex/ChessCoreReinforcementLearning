package org.leviatan.chess.engine.intel.reinforcementlearning;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.leviatan.chess.data.pgn.PGNReaderManager;
import org.leviatan.chess.data.pgn.Partida;
import org.leviatan.chess.engine.deeplearning.ConfiguracionDeepLearning;
import org.leviatan.chess.engine.intel.reinforcementlearning.network.TableroAndThreat;
import org.leviatan.chess.engine.intel.reinforcementlearning.network.ThreatEvaluation;
import org.leviatan.chess.engine.intel.reinforcementlearning.network.ThreatEvaluationLongTerm;
import org.leviatan.chess.engine.intel.reinforcementlearning.network.ThreatEvaluationNetwork;
import org.leviatan.chess.engine.intel.reinforcementlearning.network.ThreatEvaluationShortTerm;
import org.leviatan.chess.tools.platform.AppLogger;

/**
 * TestTrainerThreat.
 *
 * @author Alejandro
 *
 */
public final class TestTrainerThreat {

    private TestTrainerThreat() {
    }

    /**
     * Main method.
     *
     * @param args
     *            args
     * @throws Exception
     */
    public static void main(final String[] args) throws Exception {

        AppLogger.logDebug("Empieza el entrenamiento");

        final ThreatEvaluationNetwork threatEvaluationNetwork = new ThreatEvaluationNetwork(ConfiguracionDeepLearning.DIR_MODELOS);

        final List<Partida> listPartida = PGNReaderManager.getPartidasDelFichero(0);

        trainListPartidas(threatEvaluationNetwork, listPartida, true);
    }

    private static void trainListPartidas(final ThreatEvaluationNetwork threatEvaluationNetwork, final List<Partida> listPartida,
            final boolean shortLongTerm) throws IOException {

        final ThreatEvaluationShortTerm threatEvaluationShortTerm = new ThreatEvaluationShortTerm();
        final ThreatEvaluationLongTerm threatEvaluationLongTerm = new ThreatEvaluationLongTerm();

        final ThreatEvaluation threatEvaluation = shortLongTerm ? threatEvaluationShortTerm : threatEvaluationLongTerm;

        final int numPartidasTotal = listPartida.size();
        AppLogger.logDebug("Numero de partidas TOTAL: " + numPartidasTotal);
        int countPartidas = 0;

        for (final Partida partida : listPartida) {

            trainPartida(threatEvaluationNetwork, partida, threatEvaluation);

            countPartidas++;

            AppLogger.logDebug("Partidas entrenadas: " + countPartidas + " / " + numPartidasTotal);
        }
    }

    private static void trainPartida(final ThreatEvaluationNetwork threatEvaluationNetwork, final Partida partida,
            final ThreatEvaluation threatEvaluation) throws IOException {

        final List<TableroAndThreat> listTableroAndThreat = threatEvaluation.evaluateThreatsOfPartida(partida);

        for (final TableroAndThreat tableroAndThreat : listTableroAndThreat) {
            trainTablero(tableroAndThreat, threatEvaluationNetwork);
        }
    }

    private static void trainTablero(final TableroAndThreat tableroAndThreat, final ThreatEvaluationNetwork threatEvaluationNetwork)
            throws IOException {

        final List<TableroAndThreat> listTableroAndThreat = Arrays.asList(tableroAndThreat);
        threatEvaluationNetwork.train(10, listTableroAndThreat);
    }
}
