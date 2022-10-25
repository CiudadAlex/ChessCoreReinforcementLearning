package org.leviatan.chess.engine.intel.reinforcementlearning.network;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.leviatan.chess.data.pgn.Partida;

/**
 * @author Alejandro
 *
 */
public class ThreatEvaluationLongTerm implements ThreatEvaluation {

    private static final double ATTENUATION_FACTOR = 0.7;

    @Override
    public List<TableroAndThreat> evaluateThreatsOfPartida(final Partida partida) {

        final ThreatEvaluationShortTerm threatEvaluationShortTerm = new ThreatEvaluationShortTerm();
        final List<TableroAndThreat> listTableroAndThreat = threatEvaluationShortTerm.evaluateThreatsOfPartida(partida);

        final List<TableroAndThreat> listRev1 = cloneReversed(listTableroAndThreat);
        final List<TableroAndThreat> listRev2 = cloneReversed(listTableroAndThreat);

        for (int i = 0; i < listRev1.size(); i++) {

            final TableroAndThreat ttLast = listRev1.get(i);
            double threatAccBlancas = ttLast.getThreatBlancas() * ATTENUATION_FACTOR;
            double threatAccNegras = ttLast.getThreatNegras() * ATTENUATION_FACTOR;

            for (int k = i + 1; k < listRev2.size(); k++) {

                final TableroAndThreat tt = listRev2.get(i);
                tt.setThreatBlancasIfSuperior(threatAccBlancas);
                tt.setThreatNegraIfSuperior(threatAccNegras);

                threatAccBlancas = threatAccBlancas * ATTENUATION_FACTOR;
                threatAccNegras = threatAccNegras * ATTENUATION_FACTOR;
            }
        }

        return listTableroAndThreat;
    }

    private <T> List<T> cloneReversed(final List<T> list) {

        final List<T> listClone = new ArrayList<T>();
        listClone.addAll(list);

        Collections.reverse(listClone);

        return listClone;
    }

}
