using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Random = UnityEngine.Random;

public class DoggyAgent : Agent
{
    [Header("Сервоприводы")]
    public ArticulationBody[] legs;

    [Header("Скорость работы сервоприводов")]
    public float servoSpeed;

    [Header("Тело")]
    public ArticulationBody body;
    private Vector3 defPos;
    private Quaternion defRot;
    private float previousDistance;
    private float episodeTimer;

    [Header("Куб (цель)")]
    public GameObject cube;

    [Header("Сенсоры")]
    public Unity.MLAgentsExamples.GroundContact[] groundContacts;

    public override void Initialize()
    {
        defRot = body.transform.rotation;
        defPos = body.transform.position;
    }

    public override void OnEpisodeBegin()
    {
        episodeTimer = 0f;

        float randomY = Random.Range(0f, 360f);
        Quaternion spawnRotation = Quaternion.Euler(defRot.eulerAngles.x, randomY, defRot.eulerAngles.z);

        body.TeleportRoot(defPos, spawnRotation);
        body.velocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;

        for (int i = 0; i < legs.Length; i++)
        {
            MoveLeg(legs[i], 0);
        }

        float spawnRadius = Academy.Instance.EnvironmentParameters.GetWithDefault("cube_distance", 2.5f);
        Vector2 randomCircle = Random.insideUnitCircle.normalized * spawnRadius;
        cube.transform.position = new Vector3(randomCircle.x, 0.21f, randomCircle.y);

        previousDistance = Vector3.Distance(body.transform.position, cube.transform.position);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(body.transform.up.y);
        sensor.AddObservation(body.velocity);
        sensor.AddObservation(body.angularVelocity);

        Vector3 localTargetPos = body.transform.InverseTransformPoint(cube.transform.position);
        sensor.AddObservation(localTargetPos.normalized);
        sensor.AddObservation(localTargetPos.magnitude);

        foreach (var leg in legs)
        {
            float currentAngle = Mathf.InverseLerp(leg.xDrive.lowerLimit, leg.xDrive.upperLimit, leg.jointPosition[0]);
            sensor.AddObservation(currentAngle);
            sensor.AddObservation(leg.velocity);
        }

        foreach (var groundContact in groundContacts)
        {
            sensor.AddObservation(groundContact.touchingGround ? 1 : 0);
        }
    }

    public override void OnActionReceived(ActionBuffers vectorAction)
    {
        episodeTimer += Time.fixedDeltaTime;

        var actions = vectorAction.ContinuousActions;
        for (int i = 0; i < legs.Length; i++)
        {
            float angle = Mathf.Lerp(legs[i].xDrive.lowerLimit, legs[i].xDrive.upperLimit, (actions[i] + 1) * 0.5f);
            MoveLeg(legs[i], angle);
        }

        Vector3 toCube = cube.transform.position - body.transform.position;
        float currentDistance = toCube.magnitude;
        Vector3 dirToTarget = toCube.normalized;

        float velocityTowardCube = Vector3.Dot(body.velocity, dirToTarget);
        float lookAlignment = Vector3.Dot(body.transform.forward, dirToTarget);

        Debug.DrawRay(body.transform.position, body.transform.forward * 2, Color.red);
        Debug.DrawRay(body.transform.position, dirToTarget * 2, Color.green);

        AddReward(lookAlignment * 0.05f);

        if (Mathf.Abs(velocityTowardCube) > 0.05f)
        {
            AddReward(velocityTowardCube * 0.15f * lookAlignment);
        }

        if (currentDistance < previousDistance)
        {
            float proximityMultiplier = 1.0f / (currentDistance + 0.5f);
            AddReward(0.01f * proximityMultiplier);
            previousDistance = currentDistance;
        }

        AddReward(-0.001f);

        float currentTargetDistance = Academy.Instance.EnvironmentParameters.GetWithDefault("cube_distance", 2.5f);
        float maxTimeAllowed = 10f + (currentTargetDistance * 4f);

        if (episodeTimer > maxTimeAllowed)
        {
            AddReward(-1.0f);
            EndEpisode();
        }

        if (currentDistance < 1.0f)
        {
            AddReward(100.0f);
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        float v = Input.GetAxis("Vertical");
        for (int i = 0; i < legs.Length; i++)
        {
            continuousActionsOut[i] = v;
        }
    }

    void MoveLeg(ArticulationBody leg, float targetAngle)
    {
        leg.GetComponent<Leg>().MoveLeg(targetAngle, servoSpeed);
    }
}
