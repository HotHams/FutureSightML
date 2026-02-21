"""Neural network win probability predictor.

Architecture:
    PokemonEncoder -> TeamEncoder -> MatchupHead -> P(team1 wins)

The model processes both teams through shared encoders, then predicts
the probability of team 1 winning.
"""

import torch
import torch.nn as nn

from .embeddings import PokemonEncoder, TeamEncoder


class WinPredictor(nn.Module):
    """End-to-end neural win predictor.

    Takes two teams (each as species/moves/items/abilities indices)
    and predicts P(team 1 wins).
    """

    def __init__(
        self,
        num_species: int,
        num_moves: int,
        num_items: int,
        num_abilities: int,
        species_dim: int = 64,
        move_dim: int = 32,
        item_dim: int = 32,
        ability_dim: int = 32,
        pokemon_dim: int = 128,
        team_dim: int = 256,
        attention_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Shared encoders (both teams use the same weights)
        self.pokemon_encoder = PokemonEncoder(
            num_species=num_species,
            num_moves=num_moves,
            num_items=num_items,
            num_abilities=num_abilities,
            species_dim=species_dim,
            move_dim=move_dim,
            item_dim=item_dim,
            ability_dim=ability_dim,
            output_dim=pokemon_dim,
            dropout=dropout,
        )
        self.team_encoder = TeamEncoder(
            pokemon_dim=pokemon_dim,
            team_dim=team_dim,
            num_heads=attention_heads,
            dropout=dropout,
        )

        # Matchup head: takes both team representations + their interaction + rating features
        self.matchup_head = nn.Sequential(
            nn.Linear(team_dim * 3 + 2, team_dim),  # +2 for rating_diff, rating_avg
            nn.LayerNorm(team_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(team_dim, team_dim // 2),
            nn.LayerNorm(team_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(team_dim // 2, 1),
        )

    def forward(
        self,
        team1_species: torch.Tensor,       # (B, 6)
        team1_moves: torch.Tensor,          # (B, 6, 4)
        team1_items: torch.Tensor,          # (B, 6)
        team1_abilities: torch.Tensor,      # (B, 6)
        team2_species: torch.Tensor,        # (B, 6)
        team2_moves: torch.Tensor,          # (B, 6, 4)
        team2_items: torch.Tensor,          # (B, 6)
        team2_abilities: torch.Tensor,      # (B, 6)
        rating_features: torch.Tensor | None = None,  # (B, 2)
    ) -> torch.Tensor:
        """Predict win probability for team 1.

        Returns: (B,) tensor of probabilities in [0, 1].
        """
        # Encode Pokemon on each team
        t1_pokemon = self.pokemon_encoder(team1_species, team1_moves, team1_items, team1_abilities)
        t2_pokemon = self.pokemon_encoder(team2_species, team2_moves, team2_items, team2_abilities)

        # Create padding masks (species == 0 means empty slot)
        t1_mask = (team1_species == 0)
        t2_mask = (team2_species == 0)

        # Encode team-level synergy
        t1_repr = self.team_encoder(t1_pokemon, mask=t1_mask)  # (B, team_dim)
        t2_repr = self.team_encoder(t2_pokemon, mask=t2_mask)  # (B, team_dim)

        # Matchup features: [team1, team2, team1 - team2, rating_features]
        diff = t1_repr - t2_repr
        parts = [t1_repr, t2_repr, diff]

        if rating_features is not None:
            parts.append(rating_features)
        else:
            # Default: assume equal ratings during team building inference
            batch_size = t1_repr.size(0)
            parts.append(torch.zeros(batch_size, 2, device=t1_repr.device))

        combined = torch.cat(parts, dim=-1)  # (B, team_dim * 3 + 2)

        logits = self.matchup_head(combined).squeeze(-1)  # (B,)
        return torch.sigmoid(logits)

    def predict_team_strength(
        self,
        team_species: torch.Tensor,
        team_moves: torch.Tensor,
        team_items: torch.Tensor,
        team_abilities: torch.Tensor,
    ) -> torch.Tensor:
        """Get team representation vector (useful for team builder evaluation).

        Returns: (B, team_dim) representation.
        """
        pokemon = self.pokemon_encoder(team_species, team_moves, team_items, team_abilities)
        mask = (team_species == 0)
        return self.team_encoder(pokemon, mask=mask)
