package org.leviatan.chess.engine.intel.reinforcementlearning.training;

import org.leviatan.chess.data.pgn.PGNReaderManager;
import org.leviatan.chess.data.pgn.Partida;
import org.leviatan.chess.engine.intel.reinforcementlearning.network.*;
import org.leviatan.chess.tools.platform.AppLogger;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * TrainerThreat.
 *
 * @author Alejandro
 *
 */
public final class TrainerThreat {

    private TrainerThreat() {
    }

    public static void trainListPartidasDeFichero(final String pathModelos, final int numeroDelFile,
                                          final boolean shortLongTerm) throws Exception {

        AppLogger.logDebug("Empieza el entrenamiento");

        final ThreatEvaluationNetwork threatEvaluationNetwork = new ThreatEvaluationNetwork(pathModelos);

        final List<Partida> listPartida = PGNReaderManager.getPartidasDelFichero(numeroDelFile);

        trainListPartidas(threatEvaluationNetwork, listPartida, shortLongTerm);
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
