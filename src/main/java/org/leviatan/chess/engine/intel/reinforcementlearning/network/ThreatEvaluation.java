package org.leviatan.chess.engine.intel.reinforcementlearning.network;

import java.util.List;

import org.leviatan.chess.data.pgn.Partida;

/**
 * @author Alejandro
 *
 */
public interface ThreatEvaluation {

    public List<TableroAndThreat> evaluateThreatsOfPartida(final Partida partida);
}
