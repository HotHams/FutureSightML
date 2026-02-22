"""Neural network win probability predictor.

Architecture:
    PokemonEncoder -> TeamEncoder -> CrossAttention -> MatchupHead -> P(team1 wins)

The model processes both teams through shared encoders, applies cross-team
attention to capture matchup interactions, then predicts the probability
of team 1 winning.
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
        continuous_dim: int = 64,
        rating_dim: int = 6,
    ):
        super().__init__()
        self.pokemon_dim = pokemon_dim
        self.team_dim = team_dim
        self.rating_dim = rating_dim

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
            continuous_dim=continuous_dim,
        )
        self.team_encoder = TeamEncoder(
            pokemon_dim=pokemon_dim,
            team_dim=team_dim,
            num_heads=attention_heads,
            dropout=dropout,
        )

        # Cross-team attention: team1 Pokemon attend to team2 Pokemon (and vice versa)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=pokemon_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Project cross-attention output to matchup features
        self.cross_proj = nn.Sequential(
            nn.Linear(pokemon_dim, team_dim // 2),
            nn.LayerNorm(team_dim // 2),
            nn.GELU(),
        )

        # Matchup head: [t1_repr, t2_repr, diff, cross_diff, rating_features]
        # = team_dim * 3 + team_dim // 2 + rating_dim
        matchup_input_dim = team_dim * 3 + team_dim // 2 + rating_dim
        self.matchup_head = nn.Sequential(
            nn.Linear(matchup_input_dim, team_dim),
            nn.LayerNorm(team_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(team_dim, team_dim // 2),
            nn.LayerNorm(team_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(team_dim // 2, team_dim // 4),
            nn.LayerNorm(team_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(team_dim // 4, 1),
        )

    def _cross_attend_and_pool(
        self,
        t1_pokemon: torch.Tensor,  # (B, 6, pokemon_dim)
        t2_pokemon: torch.Tensor,  # (B, 6, pokemon_dim)
        t1_mask: torch.Tensor | None = None,  # (B, 6) True = padded
        t2_mask: torch.Tensor | None = None,  # (B, 6) True = padded
    ) -> torch.Tensor:
        """Cross-attention: t1 queries t2 and t2 queries t1, then compute diff.

        Returns: (B, pokemon_dim) cross-attention difference vector.
        """
        # t1 Pokemon attend to t2 Pokemon
        t1_cross, _ = self.cross_attn(
            query=t1_pokemon, key=t2_pokemon, value=t2_pokemon,
            key_padding_mask=t2_mask,
        )  # (B, 6, pokemon_dim)

        # t2 Pokemon attend to t1 Pokemon
        t2_cross, _ = self.cross_attn(
            query=t2_pokemon, key=t1_pokemon, value=t1_pokemon,
            key_padding_mask=t1_mask,
        )  # (B, 6, pokemon_dim)

        # Mean-pool (mask out padded slots)
        if t1_mask is not None:
            t1_cross = t1_cross.masked_fill(t1_mask.unsqueeze(-1), 0.0)
            t1_count = (~t1_mask).float().sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            t1_pooled = t1_cross.sum(dim=1) / t1_count  # (B, pokemon_dim)
        else:
            t1_pooled = t1_cross.mean(dim=1)  # (B, pokemon_dim)

        if t2_mask is not None:
            t2_cross = t2_cross.masked_fill(t2_mask.unsqueeze(-1), 0.0)
            t2_count = (~t2_mask).float().sum(dim=1, keepdim=True).clamp(min=1)
            t2_pooled = t2_cross.sum(dim=1) / t2_count
        else:
            t2_pooled = t2_cross.mean(dim=1)

        # Return difference (captures asymmetric matchup info)
        return t1_pooled - t2_pooled  # (B, pokemon_dim)

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
        rating_features: torch.Tensor | None = None,  # (B, rating_dim)
        team1_continuous: torch.Tensor | None = None,  # (B, 6, continuous_dim)
        team2_continuous: torch.Tensor | None = None,  # (B, 6, continuous_dim)
    ) -> torch.Tensor:
        """Predict win probability for team 1.

        Returns: (B,) tensor of probabilities in [0, 1].
        """
        # Encode Pokemon on each team (pass continuous features if available)
        t1_pokemon = self.pokemon_encoder(
            team1_species, team1_moves, team1_items, team1_abilities,
            continuous=team1_continuous,
        )
        t2_pokemon = self.pokemon_encoder(
            team2_species, team2_moves, team2_items, team2_abilities,
            continuous=team2_continuous,
        )

        # Create padding masks (species == 0 means empty slot)
        t1_mask = (team1_species == 0)
        t2_mask = (team2_species == 0)

        # Encode team-level synergy
        t1_repr = self.team_encoder(t1_pokemon, mask=t1_mask)  # (B, team_dim)
        t2_repr = self.team_encoder(t2_pokemon, mask=t2_mask)  # (B, team_dim)

        # Cross-team attention: capture per-Pokemon matchup interactions
        cross_diff = self._cross_attend_and_pool(t1_pokemon, t2_pokemon, t1_mask, t2_mask)
        cross_diff = self.cross_proj(cross_diff)  # (B, team_dim // 2)

        # Matchup features: [team1, team2, team1 - team2, cross_diff, rating_features]
        diff = t1_repr - t2_repr
        parts = [t1_repr, t2_repr, diff, cross_diff]

        if rating_features is not None:
            parts.append(rating_features)
        else:
            # Default: assume equal ratings during team building inference
            batch_size = t1_repr.size(0)
            parts.append(torch.zeros(batch_size, self.rating_dim, device=t1_repr.device))

        combined = torch.cat(parts, dim=-1)  # (B, team_dim * 3 + team_dim // 2 + rating_dim)

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
